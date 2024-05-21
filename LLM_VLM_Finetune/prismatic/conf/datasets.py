"""
datasets.py

Draccus Dataclass Definition for a DatasetConfig object, with various registered subclasses for each dataset variant
and processing scheme. A given dataset variant (e.g., `llava-lightning`) configures the following attributes:
    - Dataset Variant (Identifier) --> e.g., "llava-v15"
    - Align Stage Dataset Components (annotations, images)
    - Finetune Stage Dataset Components (annotations, images)
    - Dataset Root Directory (Path)
"""
from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Tuple

from draccus import ChoiceRegistry


@dataclass
class DatasetConfig(ChoiceRegistry):
    # fmt: off
    dataset_id: str                                 # Unique ID that fully specifies a dataset variant

    # Dataset Components for each Stage in < align | finetune >
    align_stage_components: Tuple[Path, Path]       # Path to annotation file and images directory for `align` stage
    finetune_stage_components: Tuple[Path, Path]    # Path to annotation file and images directory for `finetune` stage

    dataset_root_dir: Path                          # Path to dataset root directory; others paths are relative to root
    # fmt: on


######## FOR LLM DATASETS ########
@dataclass
class LLaMa2_Math_Config(DatasetConfig):
    dataset_id: str = "llama2-math"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )

    finetune_stage_components: Tuple[Path, Path] = (
        #Path("download/NumGLUE/NumGLUE_train_filtered.jsonl"),
        # Path("download/NumGLUE/DataInf_LastLayer_40_percent.jsonl"),
        # Path("download/NumGLUE/"),

        # Path("download/openbookqa/openbookqa_train.jsonl"),
        # Path("download/openbookqa/"),

        # Path("download/piqa/train_80perc.jsonl"),
        # Path("download/piqa/"),

        # Path("download/logiqa/train.jsonl"),
        # # Path("download/logiqa/random_LastLayer_40_percent.jsonl"),
        # Path("download/logiqa/"),

        Path("download/commonsenseQA/train_80perc.jsonl"),
        #Path("download/commonsenseQA/random_LastLayer_40_percent.jsonl"),
        Path("download/commonsenseQA/"),

        #Path("download/math_qa/train_NoRationale.jsonl"),
        # Path("download/math_qa/HessianFree_NoRationale_LastLayer_40_percent.jsonl"),
        # Path("download/math_qa/"),
    )

    val_components: Tuple[Path, Path] = (
        # Path("download/NumGLUE/NumGLUE_dev_filtered.jsonl"),
        # Path("download/NumGLUE/"),

        # Path("download/openbookqa/openbookqa_validation.jsonl"),
        # Path("download/openbookqa/"),

        # Path("download/piqa/train_20perc.jsonl"),
        # Path("download/piqa/"),

        # Path("download/logiqa/validation.jsonl"),
        # Path("download/logiqa/"),

        Path("download/commonsenseQA/train_20perc.jsonl"),
        Path("download/commonsenseQA/"),

    
    )
    dataset_root_dir: Path = Path("data")




# [Reproduction] LLaVa-v15 (exact dataset used in all public LLaVa-v15 models)
@dataclass
class LLaVa_V15_Config(DatasetConfig):
    dataset_id: str = "llava-v15"

    align_stage_components: Tuple[Path, Path] = (
        #Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/chat_5_percent.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/train_val/llava_v1_5_train_10_percent_HessianFree_40_perc_LLM_LastLayer.json"),
        #Path("download/llava-v1.5-instruct/train_val/llava_v1_5_train_10_percent.json"),
        #Path("download/llava-v1.5-instruct/train_val/llava_v1_5_train.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    val_components: Tuple[Path, Path] = (
        #Path("download/llava-v1.5-instruct/train_val/llava_v1_5_val.json"),
        Path("download/llava-v1.5-instruct/train_val/llava_v1_5_val.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("data")


# [Multimodal-Only] LLava-v15 WITHOUT the Language-Only ShareGPT Data (No Co-Training)
@dataclass
class LLaVa_Multimodal_Only_Config(DatasetConfig):
    dataset_id: str = "llava-multimodal"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_stripped625k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# LLaVa-v15 + LVIS-Instruct-4V
@dataclass
class LLaVa_LVIS4V_Config(DatasetConfig):
    dataset_id: str = "llava-lvis4v"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lvis4v_mix888k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# LLaVa-v15 + LRV-Instruct
@dataclass
class LLaVa_LRV_Config(DatasetConfig):
    dataset_id: str = "llava-lrv"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lrv_mix1008k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# LLaVa-v15 + LVIS-Instruct-4V + LRV-Instruct
@dataclass
class LLaVa_LVIS4V_LRV_Config(DatasetConfig):
    dataset_id: str = "llava-lvis4v-lrv"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lvis4v_lrv_mix1231k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# === Define a Dataset Registry Enum for Reference & Validation =>> all *new* datasets must be added here! ===
@unique
class DatasetRegistry(Enum):
    # === LLaMa-2 Math Dataset ===
    LLAMA2_MATH = LLaMa2_Math_Config
    # === LLaVa v1.5 ===
    LLAVA_V15 = LLaVa_V15_Config

    LLAVA_MULTIMODAL_ONLY = LLaVa_Multimodal_Only_Config

    LLAVA_LVIS4V = LLaVa_LVIS4V_Config
    LLAVA_LRV = LLaVa_LRV_Config

    LLAVA_LVIS4V_LRV = LLaVa_LVIS4V_LRV_Config

    @property
    def dataset_id(self) -> str:
        return self.value.dataset_id


# Register Datasets in Choice Registry
for dataset_variant in DatasetRegistry:
    DatasetConfig.register_subclass(dataset_variant.dataset_id, dataset_variant.value)
