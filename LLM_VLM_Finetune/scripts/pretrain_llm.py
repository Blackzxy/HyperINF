"""
pretrain.py

Pretraining script for Prismatic VLM pretraining in native PyTorch, using Fully-Sharded Data Parallel (FSDP) to run
distributed training across GPUs. By default, assumes that CUDA toolkit is >= 11.0 (to support BF16 mixed precision).


Notes & Prerequisites:
    - We're loading LLaMa-2 (and possibly other) gated models from HuggingFace (HF Hub); these require an auth_token.
      For LLaMa-2, make sure to first get Meta approval, then fill out the form at the top of the HF LLaMa-2 page:
        => Link: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
        => Generate Token (from `huggingface.co`): Settings / Access Tokens / New "Read" Token
        => Set `cfg.hf_token` to file path with token (as single line text file) or environment variable name

    - If you want to set a custom location for all HF / TIMM artifacts --> `export HF_HOME="<PATH>"` *before* running!
        => For example (add to end of .bashrc): `export HF_HOME="/mnt/fsx/skaramcheti/cache"`

Run with:
    - [Single Node One-GPU (Debug)] : torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/pretrain.py
    - [Single Node Multi-GPU (= $K)]: torchrun --standalone --nnodes 1 --nproc-per-node $K scripts/pretrain.py
    - [Multi-Node/AWS Sagemaker] Depends on your individual setup; file an issue if you have trouble!
"""
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union
import pickle
import draccus
import torch
import torch.distributed as dist
import yaml
import numpy as np

from prismatic.conf import DatasetConfig, DatasetRegistry, ModelConfig, ModelRegistry
from prismatic.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm, get_llm
from prismatic.overwatch import initialize_overwatch
from prismatic.preprocessing.materialize import get_dataset_and_collator, get_dataset_and_collator_LLM
from prismatic.training import Metrics, get_train_strategy, get_train_strategy_llm
from prismatic.util import set_global_seed

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class PretrainConfig_LLM:
    # fmt: off

    # ModelConfig (`prismatic/conf/models.py`); override with --model.type `ModelRegistry.<MODEL>.model_id`
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(ModelRegistry.Llama2_Math.model_id)
    )

    # DatasetConfig (`prismatic/conf/datasets.py`); override with --dataset.type `DatasetRegistry.<DATASET>.dataset_id`
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(DatasetRegistry.LLAMA2_MATH.dataset_id)
    )

    # Pretraining Stage in < align (projector-only) | finetune (projector + LLM) | full-finetune (all) >
    # ---
    stage: str = 'llama-dataset-pruning'# 'llama-dataset-pruning' #"llm-finetune"                                         # Pretraining Stage in < align | finetune >
    pretrained_checkpoint: Optional[Path] = None                    # Pretrained Checkpoint to Load (for `finetune`)
                                                                    #   if None =>> will match on (run_dir / `align`)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    seed: int = 7                                                   # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".simin_hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "wandb")                  # Trackers to initialize (if W&B, add config!)
    # wandb_project: str = "prismatic"                                # Name of W&B project (default: `prismatic`)
    # wandb_entity: Optional[str] = None                              # Name of W&B entity (default: None)
    wandb_project: str = "llm-PIQA"
    wandb_entity: str = "xinyu-zhou"

    def __post_init__(self) -> None:
        """Set optimization parameters based on `stage` in {"align", "finetune"}."""

        if self.stage.endswith("finetune") or self.stage == "llama-dataset-pruning" or self.stage == "inference":
            self.epochs = self.model.finetune_epochs
            self.max_steps = self.model.finetune_max_steps
            self.global_batch_size = self.model.finetune_global_batch_size
            self.per_device_batch_size = self.model.finetune_per_device_batch_size

            self.learning_rate = self.model.finetune_learning_rate
            self.weight_decay = self.model.finetune_weight_decay
            self.max_grad_norm = self.model.finetune_max_grad_norm
            self.lr_scheduler_type = self.model.finetune_lr_scheduler_type
            self.warmup_ratio = self.model.finetune_warmup_ratio

            self.train_strategy = self.model.finetune_train_strategy

        

    # fmt: on


@draccus.wrap()
def pretrain(cfg: PretrainConfig_LLM) -> None:
    overwatch.info("LLM Training :: Gathering Light")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := (overwatch.rank() % torch.cuda.device_count()))
    torch.cuda.empty_cache()

    # Create Unique Run Name & Save Directory
    model_id = cfg.model.model_id
    if (dataset_id := cfg.dataset.dataset_id) == "llava-v15":
        cfg.run_id = f"{model_id}+stage-{cfg.stage}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id
    else:
        cfg.run_id = f"{dataset_id}+{model_id}+stage-{cfg.stage}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id

    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Life is like a prism; what you see depends on how you turn the glass."', ctx_level=1)
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)
    if overwatch.is_rank_zero():
        # Additionally save a JSON version of the config
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)


    # Load LLM Backbone --> on CPU, in Full Precision (initializing Tokenizer + handling special tokens if necessary)
    overwatch.info(f"Loading Pretrained LLM [bold]{cfg.model.llm_backbone_id}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.model.llm_backbone_id, llm_max_length=cfg.model.llm_max_length, hf_token=hf_token
    )

    vision_backbone, image_transform = get_vision_backbone_and_transform(
        cfg.model.vision_backbone_id, image_resize_strategy=cfg.model.image_resize_strategy
    )

    # Create VLM => wraps `vision_backbone` and `llm`
    overwatch.info(f"Instantiating LLM `{model_id}` for Training Stage = `{cfg.stage}`")

    llm = get_llm(
        model_id,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
    )

    # for p, v in llm.llm_backbone.named_parameters():
    #     print(p, v.shape)
    # sys.exit(0)


    # [Explicit] Call to `freeze_backbones` here for clarity => will log exactly what is frozen / what's not!
    overwatch.info(f"Invoking `LLM.freeze_backbones()` for `{model_id}` => Training Stage: `{cfg.stage}`")
    llm.freeze_backbones(cfg.stage)

    # Load Weights from Checkpoint (depends on stage, config)
    overwatch.info(f"Invoking `LLM.load_checkpoint()` for `{model_id}` => Training Stage: `{cfg.stage}`")
    llm.load_from_checkpoint(cfg.stage, run_dir, pretrained_checkpoint=cfg.pretrained_checkpoint)

    # Get Dataset for Specified Stage
    overwatch.info(f"Creating Dataset `{cfg.dataset.dataset_id}` => Stage: `{cfg.stage}`")

    train_dataset, val_dataset, collator, train_jsonl, val_jsonl = get_dataset_and_collator_LLM(
        cfg.stage,
        cfg.dataset,
        tokenizer,
        prompt_builder_fn=llm_backbone.prompt_builder_fn,
        padding_side=tokenizer.padding_side,
    )

    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    train_strategy = get_train_strategy_llm(
        train_strategy=cfg.train_strategy,
        llm=llm,
        device_id=device_id,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.model.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.model.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(train_dataset))

    ####################### Data Pruning #######################
    overwatch.info("Compute Gradient")

    # val_grad_dict = train_strategy.compute_val_gradient(
    #     val_dataset,
    #     collator,
    #     seed=cfg.seed
    # )
    # with open(run_dir / "val_grad_dict.pkl", "wb") as f:
    #     pickle.dump(val_grad_dict, f)
    
    with open(run_dir / "val_grad_dict.pkl", "rb") as f:
        val_grad_dict = pickle.load(f)

    IF_dict = train_strategy.compute_training_samples_IF(
        train_dataset,
        val_grad_dict,
        collator,
        seed=cfg.seed
    )
    if_dict = IF_dict['hessian_free']
    with open(run_dir / "IF_dict.pkl", "wb") as f:
        pickle.dump(if_dict, f)
    Highquality_to_Lowquality = np.argsort(if_dict)
    with open(run_dir / "Highquality_to_Lowquality.pkl", "wb") as f:
        pickle.dump(Highquality_to_Lowquality, f)

    # with open(run_dir / "IF_dict.pkl", "rb") as f:
    #     if_dict = pickle.load(f)
    # with open(run_dir / "Highquality_to_Lowquality.pkl", "rb") as f:
    #     Highquality_to_Lowquality = pickle.load(f)

    overwatch.info("Data Pruning Done")

    n_train = len(Highquality_to_Lowquality)
    ratios = [0.05, 0.2, 0.4]
    with open(train_jsonl, "r") as f:
        data = f.readlines()

    for ratio in ratios:
        perct = int(ratio * 100)
        top_k = int(n_train * ratio)
        Highquality_to_Lowquality_topk = Highquality_to_Lowquality[:top_k]

        with open(f"data/download/commonsenseQA/hessianfree_{perct}.jsonl", "w") as f:
            for i in Highquality_to_Lowquality_topk:
                f.write(data[i])
        
    overwatch.info("Data Pruning Done")

    ####################### Data Pruning #######################


    ####################### LLM Finetune #######################
#    Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
#     overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
#     metrics = Metrics(
#         cfg.trackers,
#         cfg.run_id,
#         run_dir,
#         draccus.encode(cfg),
#         cfg.stage,
#         wandb_project=cfg.wandb_project,
#         wandb_entity=cfg.wandb_entity,
#         grad_accumulation_steps=train_strategy.grad_accumulation_steps,
#     )

#    # Run Training
#     overwatch.info("Starting Training Loop")
#     train_strategy.run_training(train_dataset, collator, metrics, stage=cfg.stage, seed=cfg.seed)

#     # Finalize
#     overwatch.info("Done with Training =>> Finalizing Metrics")
#     metrics.finalize()

#     # And... we're done!
#     overwatch.info("... and that's all, folks!")
#     dist.barrier()
#     dist.destroy_process_group()
    
    ####################### LLM Finetune #######################


if __name__ == "__main__":
    pretrain()