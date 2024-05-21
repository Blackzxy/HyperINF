from tqdm import tqdm
import pickle
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    # BitsAndBytesConfig,
    # LlamaForCausalLM,
    # LlamaTokenizer
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model
)
from datasets import Dataset
import evaluate

class LORAEngine(object):
    def __init__(self, 
                model_name_or_path="roberta-large",
                target_modules=["value"],
                train_dataloader=None,
                eval_dataloader=None,
                device="cuda",
                num_epochs=10,
                lr=3e-4,
                lora=False,
                low_rank=2,
                task="mrpc"):
        self.model_name_or_path=model_name_or_path
        self.target_modules=target_modules
        self.train_dataloader=train_dataloader
        self.eval_dataloader=eval_dataloader
        self.device=device
        self.num_epochs=num_epochs
        self.lr=lr
        self.task=task
        self.lora=lora
        self.low_rank=low_rank
        
    def build_LORA_model(self):
        '''
        This function fine-tunes a model for classification tasks. 
        For text generation tasks, please see `notebooks/Influential_Data_Identification-Llama2-Math.ipynb`.
        '''
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path,
                                                                        return_dict=True)
        self.model.config.use_cache = False
        self.model.config.pad_token_id = self.model.config.eos_token_id
            
       
        peft_config = LoraConfig(task_type="SEQ_CLS",
                                inference_mode=False, 
                                target_modules=self.target_modules,
                                r=self.low_rank,
                                lora_alpha=self.low_rank * 2,
                                use_rslora=True,
                                #lora_dropout=0.05
                                )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
   

    def train_LORA_model(self):
        '''
        This function fine-tunes a model for GLUE classification tasks. 
        For text generation tasks, please see `notebooks/Influential_Data_Identification-Llama2-Math.ipynb`.
        '''
        metric = evaluate.load("glue", self.task)
        optimizer = AdamW(params=self.model.parameters(), lr=self.lr)

        # Instantiate scheduler
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0.06*(len(self.train_dataloader)*self.num_epochs),
            num_training_steps=(len(self.train_dataloader)*self.num_epochs),
        )

        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            self.model.train()
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                batch.to(self.device)
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            self.model.eval()
            for step, batch in enumerate(tqdm(self.eval_dataloader)):
                batch.to(self.device)
                with torch.no_grad():
                    outputs = self.model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = predictions, batch["labels"]
                metric.add_batch(
                    predictions=predictions,
                    references=references,
                )

            eval_metric = metric.compute()
            print(f"Epoch {(epoch+1)}:", eval_metric)


    def compute_gradient(self, tokenized_datasets, collate_fn):
        train_dataloader_stochastic = DataLoader(tokenized_datasets["train"], 
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  batch_size=1)
        val_dataloader_stochastic = DataLoader(tokenized_datasets["validation"], 
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  batch_size=1)
        # Compute the gradient
        self.model.eval()
        tr_grad_dict = {}
        for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
            self.model.zero_grad() # zeroing out gradient
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict={}

            if self.lora:
                for k, v in self.model.named_parameters():
                    if 'lora_A' in k:
                        grad_dict[k]=v.grad.cpu()
                    elif 'lora_B' in k:
                        # first index of shape indicates low-rank
                        grad_dict[k]=v.grad.cpu().T
                    elif 'modules_to_save.default.out_proj.weight' in k:
                        grad_dict[k]=v.grad.cpu()
                    else:
                        pass
            else:
                for k, v in self.model.named_parameters():
                    ## we only choose the last layer
                    if "classifier" in k or ("23" in k and "weight" in k and "LayerNorm" not in k):
                        tmp_grad = v.grad.cpu()
                        if len(tmp_grad.shape)==1:
                            tmp_grad = tmp_grad.unsqueeze(0)
                        grad_dict[k]=tmp_grad
                        del tmp_grad, v.grad

            tr_grad_dict[step]=grad_dict
            del grad_dict
            
        val_grad_dict = {}
        for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
            self.model.zero_grad() # zeroing out gradient
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict={}

            if self.lora:
                for k, v in self.model.named_parameters():
                    if 'lora_A' in k:
                        grad_dict[k]=v.grad.cpu()
                    elif 'lora_B' in k:
                        # first index of shape indicates low-rank
                        grad_dict[k]=v.grad.cpu().T
                    elif 'modules_to_save.default.out_proj.weight' in k:
                        grad_dict[k]=v.grad.cpu()
                    else:
                        pass
            
            else:
                for k, v in self.model.named_parameters():
                    ## we only choose the last layer
                    if "classifier" in k or ("23" in k and "weight" in k and "LayerNorm" not in k):
                        tmp_grad = v.grad.cpu()
                        if len(tmp_grad.shape)==1:
                            tmp_grad = tmp_grad.unsqueeze(0)
                        grad_dict[k]=tmp_grad
                        del tmp_grad, v.grad



            val_grad_dict[step]=grad_dict    
            del grad_dict
            
        return tr_grad_dict, val_grad_dict


