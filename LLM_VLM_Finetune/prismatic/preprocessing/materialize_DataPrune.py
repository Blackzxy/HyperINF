"""
materialize.py

Factory class for initializing pretraining datasets on a per-VLM basis; provides and exports individual functions for
clear control flow.
"""
from typing import Tuple, Type, Callable

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.conf import DatasetConfig
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.preprocessing.datasets import AlignDataset, FinetuneDataset, TextVQATaskDataset
from prismatic.util.data_utils import PaddedCollatorForLanguageModeling

# Dataset Initializers =>> Maps Stage --> cls()
DATASET_INITIALIZER = {"align": AlignDataset, 
                       "finetune": FinetuneDataset, 
                       "full-finetune": FinetuneDataset, 
                       "data-pruning_projector": FinetuneDataset,
                       "data-pruning_llm": FinetuneDataset,
                       "text-vqa": TextVQATaskDataset,
                       }


def get_dataset_and_collator(
    stage: str,
    dataset_cfg: DatasetConfig,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
) -> Tuple[Dataset, Dataset, PaddedCollatorForLanguageModeling]:
    train_dataset_cls = DATASET_INITIALIZER[stage]
    val_dataset_cls = DATASET_INITIALIZER[stage]
    

    dataset_root_dir = dataset_cfg.dataset_root_dir


    collator = PaddedCollatorForLanguageModeling(
        tokenizer.model_max_length, tokenizer.pad_token_id, default_image_resolution, padding_side=padding_side
    )


    train_annotation_json, train_image_dir = dataset_cfg.finetune_stage_components ## training data
    val_annotation_json, val_image_dir = dataset_cfg.val_components ## validation data
    # annotation_json: llava_v1_5_mix665k.json, image_dir: llava-v1.5-instruct/
    
    

    train_dataset = train_dataset_cls(
        dataset_root_dir / train_annotation_json,
        dataset_root_dir / train_image_dir,
        image_transform,
        tokenizer,
        prompt_builder_fn=prompt_builder_fn,
    )

    val_dataset = val_dataset_cls(
        dataset_root_dir / val_annotation_json,
        dataset_root_dir / val_image_dir,
        image_transform,
        tokenizer,
        prompt_builder_fn=prompt_builder_fn,
    )



    return train_dataset, val_dataset, collator, dataset_root_dir/train_annotation_json, dataset_root_dir/val_annotation_json

 