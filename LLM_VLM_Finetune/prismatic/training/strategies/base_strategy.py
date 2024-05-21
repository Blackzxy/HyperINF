"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional
import copy
import numpy as np
import pandas as pd
import gc

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms import PrismaticVLM, PrismaticLLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForLanguageModeling, LLMPaddedCollatorForLanguageModeling

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        **_: str,
    ) -> None:
        self.vlm, self.device_id = vlm, device_id

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-Device Batch Size must evenly divide Global Batch Size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                            return

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()


    def compute_val_gradient(
        self,
        val_dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
        stage: str = "data-pruning_llm",
    ) -> None:

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collator,
            num_workers=32,
            worker_init_fn=self.worker_init_fn,
        )


        # === Processing ===
    
        self.vlm.eval()
                    
        val_grad_dict_avg = {}
        val_grads_list = []
        cnt = 0
        for val_idx, batch in enumerate(tqdm(val_dataloader)):
            # [Contract] self.vlm.forward() must automatically compute `loss` and return!
            
            
            self.vlm.zero_grad(set_to_none=True)
            skip_fg = True
            output: CausalLMOutputWithPast = self.vlm(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                multimodal_indices=batch["multimodal_indices"],
            )
            loss = output.loss
            if loss.requires_grad == False:
                loss.requires_grad_(True)
            loss.backward()
            

            #### only use projector gradients
            if stage == "data-pruning_projector":
                for k,v in self.vlm.projector.named_parameters():
                    ## check if v.grad is None
                    if v.grad is not None:
                        tmp_grad = v.grad.detach()
                        if len(tmp_grad.shape)==1:
                            tmp_grad = tmp_grad.unsqueeze(0)


                        if k not in val_grad_dict_avg:
                            val_grad_dict_avg[k] = tmp_grad
                        else:
                            val_grad_dict_avg[k] += tmp_grad
                        skip_fg = False
                        del v.grad, tmp_grad

            ### use the last layer gradients
            else:
                for k,v, in self.vlm.llm_backbone.named_parameters():
                    if v.grad is not None:

                        if "31" in k:
                            tmp_grad = v.grad.detach()
                            if len(tmp_grad.shape)==1:
                                tmp_grad = tmp_grad.unsqueeze(0)
                                
                            if k not in val_grad_dict_avg:
                                val_grad_dict_avg[k] = tmp_grad
                            else:
                                val_grad_dict_avg[k] += tmp_grad
                            skip_fg = False
                            del v.grad, tmp_grad
                        
            
            if skip_fg:
                continue

            cnt+=1

            del output, loss
            torch.cuda.empty_cache()

        ## avg the gradients of each layer
        for k,v in val_grad_dict_avg.items():
            val_grad_dict_avg[k] = v/cnt

        return val_grad_dict_avg
    
    def compute_training_samples_IF(
        self,
        train_dataset: Dataset,
        val_grad_dict_avg: dict,
        collator: PaddedCollatorForLanguageModeling,
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
        stage: str = "data-pruning_llm",
        method: str = "hyperinf"
    ) -> None:

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collator,
            num_workers=32,
            worker_init_fn=self.worker_init_fn,
        )


        def schulz_inverse_stable(A, damping_factor=0, max_iterations=20, tol=1e-6):
            n = A.shape[0]
            I = torch.eye(n, device=A.device)
            A_damped = A + damping_factor * I  # Apply damping

            #X = np.eye(n) * 0.00001  # Initial estimate of inverse matrix
            X = torch.eye(n, device=A.device) * 0.00005  # Initial estimate of inverse matrix

            for _ in range(max_iterations):
                #X = X.dot(2 * I - A_damped.dot(X))
                X = X @ (2 * I - A_damped @ X)
            return X


        # === Processing ===
    
        self.vlm.eval()
                    
        train_grad_dict = {}
        cnt = 0
        lambda_l = {} ## lambda for each layer
        G_l = {} ## G for each layer
        G_l_inv = {} ## G inverse for each layer
        lambda_const = 10
        hvp_dict = {}
        ratio_l = {}

        ## compute the lambda_l for each layer first for DATAINF method
        if method == "datainf":
            for tr_idx, batch in enumerate(tqdm(train_dataloader)):
            
                self.vlm.zero_grad(set_to_none=True)
                skip_fg = True
                output: CausalLMOutputWithPast = self.vlm(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                    multimodal_indices=batch["multimodal_indices"],
                )
                loss = output.loss
                if loss.requires_grad == False:
                    loss.requires_grad_(True)
                loss.backward()
                

                if stage == "data-pruning_projector":
                    for k,v in self.vlm.projector.named_parameters():
                        ## check if v.grad is None
                        if v.grad is not None:
                            tmp_grad = v.grad.detach()
                            if len(tmp_grad.shape)==1:
                                    tmp_grad = tmp_grad.unsqueeze(0)

                            if k not in lambda_l:
                                lambda_l[k] = torch.mean(tmp_grad**2)
                            else:
                                lambda_l[k] += torch.mean(tmp_grad**2)
                            
                            skip_fg = False
                            del v.grad, tmp_grad
                else:
                    for k,v, in self.vlm.llm_backbone.named_parameters():
                        if v.grad is not None:

                            if "31" in k:
                                tmp_grad = v.grad.detach()
                                if len(tmp_grad.shape)==1:
                                    tmp_grad = tmp_grad.unsqueeze(0)
                                    
                                if k not in lambda_l:
                                    lambda_l[k] = torch.mean(tmp_grad**2)
                                else:
                                    lambda_l[k] += torch.mean(tmp_grad**2)
                                skip_fg = False
                                del v.grad, tmp_grad
                
                if skip_fg:
                    continue

                
                cnt+=1
                del output, loss
                torch.cuda.empty_cache()
            
            for weight_name in lambda_l.keys():
                lambda_l[weight_name] = lambda_l[weight_name]/cnt/lambda_const
        

            
        


        for tr_idx, batch in enumerate(tqdm(train_dataloader)):
            
           
            self.vlm.zero_grad(set_to_none=True)
            skip_fg = True
            output: CausalLMOutputWithPast = self.vlm(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                multimodal_indices=batch["multimodal_indices"],
            )
            loss = output.loss
            if loss.requires_grad == False:
                loss.requires_grad_(True)
            loss.backward()
            
            ## only use the projector gradients
            if stage == "data-pruning_projector":
                for k,v in self.vlm.projector.named_parameters():
                    ## check if v.grad is None
                    if v.grad is not None:
                        tmp_grad = v.grad.detach()
                        if len(tmp_grad.shape)==1:
                            tmp_grad = tmp_grad.unsqueeze(0)

                        if method == "hyperinf" or method == "lissa":
                            if k not in G_l:
                                G_l[k] = tmp_grad.T @ tmp_grad
                            else:
                                G_l[k] += tmp_grad.T @ tmp_grad
                            
                            if k not in lambda_l:
                                lambda_l[k] = torch.mean(tmp_grad**2)
                            else:
                                lambda_l[k] += torch.mean(tmp_grad**2)
                        
                        
                        elif method == "datainf":
                            if k not in ratio_l:
                                ratio_l[k] = (tmp_grad.T @ tmp_grad) / (lambda_l[k] + torch.sum(tmp_grad**2))
                            else:
                                ratio_l[k] += (tmp_grad.T @ tmp_grad) / (lambda_l[k] + torch.sum(tmp_grad**2))
                        
                        skip_fg = False
                        del v.grad, tmp_grad
                

            ### use the last layer gradients
            elif stage == "data-pruning_llm":
                for k,v in self.vlm.llm_backbone.named_parameters():
                    if v.grad is not None:
                        
                        if '31' in k:
                            tmp_grad = v.grad.detach()
                            if len(tmp_grad.shape)==1:
                                tmp_grad = tmp_grad.unsqueeze(0)
                            
                            if method == "hyperinf" or method == "lissa":
                                if k not in G_l:
                                    G_l[k] = tmp_grad.T @ tmp_grad
                                else:
                                    G_l[k] += tmp_grad.T @ tmp_grad
                                
                                if k not in lambda_l:
                                    lambda_l[k] = torch.mean(tmp_grad**2)
                                else:
                                    lambda_l[k] += torch.mean(tmp_grad**2)

                            elif method == "datainf":
                                if k not in ratio_l:
                                    ratio_l[k] = (tmp_grad.T @ tmp_grad) / (lambda_l[k] + torch.sum(tmp_grad**2))
                                else:
                                    ratio_l[k] += (tmp_grad.T @ tmp_grad) / (lambda_l[k] + torch.sum(tmp_grad**2))
                            
                        
                            skip_fg = False
                            del v.grad, tmp_grad
            
            if skip_fg:
                continue
            

            cnt+=1


            del output, loss
            torch.cuda.empty_cache()

            # ## save tr_idx, cnt, G_l, lambda_l
            if cnt%1000==0:
                np.save('tr_idx.npy', tr_idx)
                np.save('cnt.npy', cnt)
                np.save('G_l.npy', G_l)
                np.save('lambda_l.npy', lambda_l)
                # np.save('ratio_l.npy', ratio_l)
        

        ####################### HyperINF method ############################
        if method == "hyperinf":
            for weight_name in lambda_l.keys():
                lambda_l[weight_name] = lambda_l[weight_name]/cnt/lambda_const
            
            
            hvp_iter_dict = {}
            for weight_name in G_l.keys():
                G_l[weight_name] = G_l[weight_name]/cnt + lambda_l[weight_name]*torch.eye(G_l[weight_name].shape[0], device=G_l[weight_name].device)
                G_l_inv[weight_name] = schulz_inverse_stable(G_l[weight_name], damping_factor=0, max_iterations=15, tol=1e-6)
            
            for weight_name in G_l_inv.keys():
                hvp_iter_dict[weight_name] = val_grad_dict_avg[weight_name] @ G_l_inv[weight_name]
            
            hvp_dict['hyperinf'] = hvp_iter_dict
            del G_l, G_l_inv, hvp_iter_dict
            gc.collect()

            # save hvp_dict
            np.save('hvp_dict.npy', hvp_dict)
        
        ###################### LISSA method ############################
        if method == "lissa":
            for weight_name in lambda_l.keys():
                lambda_l[weight_name] = lambda_l[weight_name]/cnt/lambda_const

            hvp_lissa_dict = {}
            for weight_name in G_l.keys():
                G_l[weight_name] = G_l[weight_name]/cnt + lambda_l[weight_name]*torch.eye(G_l[weight_name].shape[0], device=G_l[weight_name].device)
                N_iter = 10
                r_l = val_grad_dict_avg[weight_name]
                for _ in range(N_iter):
                    r_l = val_grad_dict_avg[weight_name] + r_l @ (torch.eye(G_l[weight_name].shape[0], device=G_l[weight_name].device) - G_l[weight_name])
                hvp_lissa_dict[weight_name] = r_l
                del r_l, N_iter

            hvp_dict['lissa'] = hvp_lissa_dict
            del G_l, hvp_lissa_dict

            np.save('hvp_dict.npy', hvp_dict)
        

        # ######################## DATAINF method ############################
        if method == "datainf":
            hvp_datainf_dict = {}
            for weight_name in ratio_l.keys():
                hvp_datainf_dict[weight_name] = 1 / lambda_l[weight_name] * (val_grad_dict_avg[weight_name] - val_grad_dict_avg[weight_name] @ ratio_l[weight_name] / cnt)
            hvp_dict['datainf'] = hvp_datainf_dict

            del ratio_l, hvp_datainf_dict
            gc.collect()
            np.save('hvp_dict.npy', hvp_dict)

        # ######################## DATAINF method ############################

        ########################## Hessian-Free method ############################
        if method == "hessian_free":
            hvp_dict['hessian_free'] = val_grad_dict_avg
            del val_grad_dict_avg
            gc.collect()
        ########################## Hessian-Free method ############################
        



        ########################### compute IF score ############################
        
        IF_dict = {}
        for mtd in hvp_dict.keys():
            if_tmp_dict = {}
            cnt=0
            for tr_id, batch in enumerate(tqdm(train_dataloader)): # iterate over the training dataset
                
                self.vlm.zero_grad(set_to_none=True)
                skip_fg = True
                output: CausalLMOutputWithPast = self.vlm(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                    multimodal_indices=batch["multimodal_indices"],
                )
                loss = output.loss
                if loss.requires_grad == False:
                    loss.requires_grad_(True)
                loss.backward()
                
                ## we only store the gradients of the trainable modules
                if_tmp_score = 0

                ## only use the projector gradients
                if stage == "data-pruning_projector":
                    for k,v in self.vlm.projector.named_parameters():
                        ## check if v.grad is None
                        if v.grad is not None:
                            tmp_grad = v.grad.detach()
                            if len(tmp_grad.shape)==1:
                                tmp_grad = tmp_grad.unsqueeze(0)
                            if_tmp_score += torch.sum(hvp_dict[mtd][k] * tmp_grad)
                            skip_fg = False
                            del v.grad, tmp_grad


                ## use the last layer gradients
                else:
                    for k,v in self.vlm.llm_backbone.named_parameters():
                        if v.grad is not None:
                        
                            if '31' in k:
                                tmp_grad = v.grad.detach()
                                if len(tmp_grad.shape)==1:
                                    tmp_grad = tmp_grad.unsqueeze(0)
                                if_tmp_score += torch.sum(hvp_dict[mtd][k] * tmp_grad)
                                skip_fg = False
                                del v.grad, tmp_grad

                        
                
                if skip_fg:
                    continue
                


                cnt+=1

                print(-if_tmp_score)
                if_tmp_dict[tr_id] = -if_tmp_score.cpu().detach().numpy()

                
                del output, loss                
                torch.cuda.empty_cache()

                if tr_id % 1000 == 0:
                    ## save tr_id, if_tmp_dict
                    np.save('tr_id.npy', tr_id)
                    np.save('if_tmp_dict.npy', if_tmp_dict)
            
            IF_dict[mtd] = pd.Series(if_tmp_dict, dtype=float).to_numpy()
        
        return IF_dict
        

class TrainingStrategyLLM(ABC):
    def __init__(
        self,
        llm: PrismaticLLM,
        device_id: int,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        **_: str,
    ) -> None:
        self.llm, self.device_id = llm, device_id

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.llm.all_module_keys, self.llm.trainable_module_keys
        self.llm_transformer_layer_cls = self.llm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-Device Batch Size must evenly divide Global Batch Size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    def run_training(
        self,
        dataset: Dataset,
        collator: LLMPaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        
        sampler = DistributedSampler(
            dataset,
            num_replicas=overwatch.world_size(),
            rank=overwatch.rank(),
            shuffle=True,
            seed=seed,
            drop_last=False,
        )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.llm.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.llm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )
                        loss = output.loss

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                            return

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()


    def compute_val_gradient(
        self,
        val_dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collator,
            num_workers=32,
            worker_init_fn=self.worker_init_fn,
        )


        # === Processing ===
    
        self.llm.eval()
                    
        val_grad_dict_avg = {}
        val_grads_list = []
        cnt = 0
        for val_idx, batch in enumerate(tqdm(val_dataloader)):
            # [Contract] self.vlm.forward() must automatically compute `loss` and return!
            
            
            self.llm.zero_grad(set_to_none=True)
            skip_fg = True
            output: CausalLMOutputWithPast = self.llm(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = output.loss
            if loss.requires_grad == False:
                loss.requires_grad_(True)
            loss.backward()
            

    

            for k,v, in self.llm.llm_backbone.named_parameters():
                
                if "layers.31" in k:
                    #if "31" in k:
                    tmp_grad = v.grad.detach()
                    if len(tmp_grad.shape)==1:
                        tmp_grad = tmp_grad.unsqueeze(0)
                        
                    if k not in val_grad_dict_avg:
                        val_grad_dict_avg[k] = tmp_grad
                    else:
                        val_grad_dict_avg[k] += tmp_grad
                
                    del v.grad, tmp_grad
                            

            
            cnt+=1

            del output, loss
            torch.cuda.empty_cache()
            gc.collect()

        ## avg the gradients of each layer
        for k,v in val_grad_dict_avg.items():
            val_grad_dict_avg[k] = v/cnt

        return val_grad_dict_avg
    
    def compute_training_samples_IF(
        self,
        train_dataset: Dataset,
        val_grad_dict_avg: dict,
        collator: PaddedCollatorForLanguageModeling,
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
        method: str = "hyperinf"
    ) -> None:

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collator,
            num_workers=32,
            worker_init_fn=self.worker_init_fn,
        )


        def schulz_inverse_stable(A, damping_factor=0, max_iterations=20, tol=1e-6):
            n = A.shape[0]
            I = torch.eye(n, device=A.device)
            A_damped = A + damping_factor * I  # Apply damping

            #X = np.eye(n) * 0.00001  # Initial estimate of inverse matrix
            X = torch.eye(n, device=A.device) * 0.00005  # Initial estimate of inverse matrix

            for _ in range(max_iterations):
                #X = X.dot(2 * I - A_damped.dot(X))
                X = X @ (2 * I - A_damped @ X)
            return X


        # === Processing ===
    
        self.llm.eval()
                    
        train_grad_dict = {}
        cnt = 0
        lambda_l = {} ## lambda for each layer
        G_l = {} ## G for each layer
        G_l_inv = {} ## G inverse for each layer
        lambda_const = 10
        hvp_dict = {}
        ratio_l = {}

        ## compute the lambda_l for each layer first for DATAINF method
        if method == "datainf":
            for tr_idx, batch in enumerate(tqdm(train_dataloader)):
            
                self.llm.zero_grad(set_to_none=True)
                skip_fg = True
                output: CausalLMOutputWithPast = self.llm(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = output.loss
                if loss.requires_grad == False:
                    loss.requires_grad_(True)
                loss.backward()
                

                for k,v, in self.llm.llm_backbone.named_parameters():
                    if v.grad is not None:
        
                        if "layers.31" in k:
                            tmp_grad = v.grad.detach()
                            if len(tmp_grad.shape)==1:
                                tmp_grad = tmp_grad.unsqueeze(0)
                                
                            if k not in lambda_l:
                                lambda_l[k] = torch.mean(tmp_grad**2)
                            else:
                                lambda_l[k] += torch.mean(tmp_grad**2)
                            skip_fg = False
                            del v.grad, tmp_grad
                
                if skip_fg:
                    continue

                cnt+=1
                del output, loss
                torch.cuda.empty_cache()
            
            for weight_name in lambda_l.keys():
                lambda_l[weight_name] = lambda_l[weight_name]/cnt/lambda_const
            

        cnt=0

        for tr_idx, batch in enumerate(tqdm(train_dataloader)):
            # [Contract] self.vlm.forward() must automatically compute `loss` and return!
            
           
            self.llm.zero_grad(set_to_none=True)
            skip_fg = True
            output: CausalLMOutputWithPast = self.llm(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = output.loss
            if loss.requires_grad == False:
                loss.requires_grad_(True)
            loss.backward()
            

            for k,v in self.llm.llm_backbone.named_parameters():
            
                if "layers.31" in k:
                    tmp_grad = v.grad.detach()
                    del v.grad
                    if len(tmp_grad.shape)==1:
                        tmp_grad = tmp_grad.unsqueeze(0)
                    
                    if method == "hyperinf" or method == "lissa":
                        if k not in G_l:
                            G_l[k] = tmp_grad.T @ tmp_grad
                        else:
                            G_l[k] += tmp_grad.T @ tmp_grad
                        
                        if k not in lambda_l:
                            lambda_l[k] = torch.mean(tmp_grad**2)
                        else:
                            lambda_l[k] += torch.mean(tmp_grad**2)
                   

                    elif method == "datainf":
                        if k not in ratio_l:
                            ratio_l[k] = (tmp_grad.T @ tmp_grad) / (lambda_l[k] + torch.sum(tmp_grad**2))
                        else:
                            ratio_l[k] += (tmp_grad.T @ tmp_grad) / (lambda_l[k] + torch.sum(tmp_grad**2))
                        

                    del tmp_grad
            
            cnt+=1


            del output, loss
            torch.cuda.empty_cache()
        

        ####################### HyperINF method ############################
        if method == "hyperinf":
            for weight_name in lambda_l.keys():
                lambda_l[weight_name] = lambda_l[weight_name]/cnt/lambda_const
            
            hvp_iter_dict = {}
            for weight_name in G_l.keys():
                G_l[weight_name] = G_l[weight_name]/cnt + lambda_l[weight_name]*torch.eye(G_l[weight_name].shape[0], device=G_l[weight_name].device)
                G_l_inv[weight_name] = schulz_inverse_stable(G_l[weight_name], damping_factor=0, max_iterations=30, tol=1e-6)
            
            for weight_name in G_l_inv.keys():
                hvp_iter_dict[weight_name] = val_grad_dict_avg[weight_name] @ G_l_inv[weight_name]
            
            hvp_dict['hyperinf'] = hvp_iter_dict
            del G_l, G_l_inv, hvp_iter_dict
            gc.collect()
        
        ###################### LISSA method ############################
        if method == "lissa":
            for weight_name in lambda_l.keys():
                lambda_l[weight_name] = lambda_l[weight_name]/cnt/lambda_const

            hvp_lissa_dict = {}
            for weight_name in G_l.keys():
                G_l[weight_name] = G_l[weight_name]/cnt + lambda_l[weight_name]*torch.eye(G_l[weight_name].shape[0], device=G_l[weight_name].device)
                N_iter = 10
                r_l = val_grad_dict_avg[weight_name]
                for _ in range(N_iter):
                    r_l = val_grad_dict_avg[weight_name] + r_l @ (torch.eye(G_l[weight_name].shape[0], device=G_l[weight_name].device) - G_l[weight_name])
                hvp_lissa_dict[weight_name] = r_l
                del r_l, N_iter
                

            hvp_dict['lissa'] = hvp_lissa_dict
            del G_l, hvp_lissa_dict
        


        # ######################## DATAINF method ############################
        if method == "datainf":
            hvp_datainf_dict = {}
            for weight_name in ratio_l.keys():
                hvp_datainf_dict[weight_name] = 1 / lambda_l[weight_name] * (val_grad_dict_avg[weight_name] - val_grad_dict_avg[weight_name] @ ratio_l[weight_name] / cnt)
            hvp_dict['datainf'] = hvp_datainf_dict

            del ratio_l, hvp_datainf_dict
            gc.collect()
        # ######################## DATAINF method ############################


        ########################## Hessian-Free method ############################
        if method == "hessian_free":
            hvp_dict['hessian_free'] = val_grad_dict_avg
            del val_grad_dict_avg
            gc.collect()
        ########################## Hessian-Free method ############################
        



        ########################### compute IF score ############################
        
        IF_dict = {}
        for mtd in hvp_dict.keys():
            if_tmp_dict = {}
            cnt=0
            for tr_id, batch in enumerate(tqdm(train_dataloader)): # iterate over the training dataset
                
                self.llm.zero_grad(set_to_none=True)
                skip_fg = True
                output: CausalLMOutputWithPast = self.llm(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = output.loss
                if loss.requires_grad == False:
                    loss.requires_grad_(True)
                loss.backward()
                
                ## we only store the gradients of the trainable modules
                if_tmp_score = 0

                for k,v in self.llm.llm_backbone.named_parameters():
                  
                    if  "layers.31" in k:
                        tmp_grad = v.grad.detach()
                        if len(tmp_grad.shape)==1:
                            tmp_grad = tmp_grad.unsqueeze(0)
                        if_tmp_score += torch.sum(hvp_dict[mtd][k] * tmp_grad)
                        skip_fg = False
                        del v.grad, tmp_grad

                    

                cnt+=1

                print(-if_tmp_score)
                if_tmp_dict[tr_id] = -if_tmp_score.cpu().detach().numpy()

                
                del output, loss                
                torch.cuda.empty_cache()
            
            IF_dict[mtd] = pd.Series(if_tmp_dict, dtype=float).to_numpy()
        
        return IF_dict
        

        