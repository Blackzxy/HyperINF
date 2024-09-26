"""
datasets.py

PyTorch Dataset Definitions for Prismatic models; supports processing for both the `align` and `finetune` stages, with
utilities for formatting conversations during the `finetune` stage subject to the given LLM backbone's expected
formatting (e.g., SYS_PROMPT + USER: ... ASSISTANT: ... for Vicuña v1.5 Chat models).

We currently only support Map-style Datasets; assumes that all files (annotations, images) are on local disk, and that
random access image reading is relatively cheap/fast.
"""

# from pathlib import Path
# import sys
# sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent.parent))
import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple, Type, Callable, Optional
import numpy as np
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from transformers import LlamaTokenizerFast, PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.textvqa_m4c_evaluators import EvalAIAnswerProcessor

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class AlignDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        chat_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__()
        self.chat_json, self.image_dir = chat_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.dataset_type = "align"

        # Create Prompt Template
        self.prompt_template = "{caption}" + self.tokenizer.eos_token

        # Load Chat JSON
        with open(self.chat_json, "r") as f:
            self.examples = json.load(f)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Following the *actual* code executed from the LLaVa codebase, during the "align" phase, we actually discard
        the "prompt" from the human, and instead directly predict the caption from the image.

        As a concrete example given the "raw data" for the first example:
            example = self.examples[0]["conversations]` = {
                [
                    {"from": "human", "value": "Render a clear and concise summary of the photo.\n<image>"},
                    {"from": "gpt", "value": "select luxury furniture 3 - inch gel memory foam mattress topper"}
                ]
            }

        Return =>> self.tokenizer("<image> select luxury furniture 3 - inch gel memory foam mattress topper\n")

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        image_path, conversation = Path(self.examples[idx]["image"]), self.examples[idx]["conversations"]
        assert (len(conversation) == 2) and ("<image>" not in conversation[-1]["value"]), "Unexpected text!"

        # Format Caption --> {caption}{eos_token}
        caption = self.prompt_template.format(caption=conversation[-1]["value"].strip())

        # We treat image patches as "tokens = [p1 p2 p3, ...]"; we need to specify ordering of text/patch tokens.
        #   => Critically, we find that inserting *after* the BOS token leads to the strongest performance!
        #       - input_ids = "<s> p1 p2 p3 ... <caption_text> \n"
        #       - labels = "IGNORE IGNORE ..." (copy `input_ids` replacing <s> and p{1...K} with IGNORE)
        #
        # IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids = self.tokenizer(caption, truncation=True, return_tensors="pt").input_ids[0]
        labels = copy.deepcopy(input_ids)

        # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
        labels[0] = IGNORE_INDEX

        # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
        pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert("RGB"))

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

    def get_modality_lengths(self, n_image_patches: int) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example
            n_words = sum([len(turn["value"].replace("<image>", "").split()) for turn in example["conversations"]])
            modality_lengths.append((is_multimodal, (n_image_patches + n_words) if is_multimodal else n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)

class TextVQATaskDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self, 
        root_dir: Path,
        index_file: Path,
        prompt_fn: Callable[[str], str],
        image_processor=None,
        tokenizer=None,
    )-> None:
        self.prompt_fn, self.image_processor = prompt_fn, image_processor
        self.root_dir, self.index_file = root_dir, index_file
        self.tokenizer = tokenizer
        self.answer_processor = EvalAIAnswerProcessor()

        # Load Index File
        with open(self.index_file, "r") as f:
            self.examples = list(json.load(f).values())
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        qprompt_ocr = self.prompt_fn(ex["question"])
        qprompt_no_ocr = self.prompt_fn(ex["question"].split("\nReference OCR token:")[0])

        answers = [self.answer_processor(a) for a in ex["answers"]]
        assert len(answers) == 10, f"Expected 10 answers, got {len(answers)}"
        gt_answers = list(enumerate(answers))
        unique_answers = set(answers)
        ##NOTE: randomly choose one as the ground truth
        idx = np.random.choice(len(unique_answers))
        gt_answer = list(unique_answers)[idx]
        ##NOTE: randomly choose one as the ground truth


        if isinstance(self.image_processor, Compose) or hasattr(self.image_processor, "is_prismatic"):
            # This is a standard `torchvision.transforms` object or custom PrismaticVLM wrapper
            pixel_values = self.image_processor(Image.open(self.root_dir / ex["img_path"]).convert("RGB"))

        else:
            # Assume `image_transform` is an HF ImageProcessor...
            pixel_values = self.image_processor(
                Image.open(self.root_dir / ex["img_path"]).convert("RGB"), return_tensors="pt"
            )["pixel_values"][0]
        
        labels = self.tokenizer(gt_answer).input_ids
        ##NOTE: we take qprompt_no_ocr as the input
        input_ids = self.tokenizer(qprompt_no_ocr).input_ids
        ##NOTE: we take qprompt_no_ocr as the input

        ## concatenate the input_ids (list) and labels (list)
        input_ids = input_ids + labels
        labels = [IGNORE_INDEX for _ in range(len(input_ids) - len(labels))] + labels

        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        
        ## handle max length
        input_ids, labels = input_ids[: self.tokenizer.model_max_length], labels[: self.tokenizer.model_max_length]

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
    
    def __len__(self) -> int:
        return len(self.examples)
        
class LLMFinetuneDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        is_jsonl: bool,
        instruct_jsonl: Path,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        super().__init__()
        self.instruct_jsonl = instruct_jsonl
        self.tokenizer = tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset_type = "finetune"
        self.is_jsonl = is_jsonl

        # # Load jsonl file for QA dataset
        if self.is_jsonl:
            with open(self.instruct_jsonl, "r") as f:
                self.examples = [json.loads(line) for line in f]

        ## Load JSON file for oasst1 dataset
        else:
            with open(self.instruct_jsonl, "r") as f:
                self.examples = json.load(f)


    # === Unimodal + Multimodal Handling ===
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Unlike the *align* stage handling, for the *finetune* stage, we actually need to handle multiple "turns" of
        dialog grounded in a single image.

        To do this, we leverage the `prompt_builder_fn` which instantiates a PromptBuilder object. By calling the
        methods for adding turns and getting a prompt, we ensure proper formatting and consistency for each example.

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """

        # ### For QA dataset format
        if self.is_jsonl:
            if "passage" in self.examples[idx]:
                passage = self.examples[idx]["passage"]
                question = self.examples[idx]["question"]
                question = passage + "\n" + question
                answer = str(self.examples[idx]["answer"]["number"])
                # type_q = self.examples[idx]["type"]

            else:
                question = self.examples[idx]["question"]
                answer = str(self.examples[idx]["answer"])
                # if "type" in self.examples[idx]:
                #     type_q = self.examples[idx]["type"]

        

            # Create Prompt Builder --> add each message sequentially
            prompt_builder, input_ids, labels = self.prompt_builder_fn(model_family="prismatic"), [], []

            msg = question + "\n" + answer

            #prompt_builder.get_potential_prompt(question + "\n" + answer)
            if isinstance(self.tokenizer, LlamaTokenizerFast):
                    msg = msg.rstrip()
            else:
                raise ValueError(f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!")
            
            input_ids = self.tokenizer(msg, add_special_tokens=True).input_ids
            labels = input_ids
            ### For QA dataset format

        # ### For oasst1 dataset format
        else:
            conversation = self.examples[idx]["conversations"]
            # conversation_cp = copy.deepcopy(conversation)
            # conversation = None

            # Create Prompt Builder --> add each message sequentially
            prompt_builder, input_ids, labels = self.prompt_builder_fn(model_family="prismatic"), [], []
            for turn_idx, turn in enumerate(conversation):
                # Get "effective" string added to prompt --> handle whitespace for tokenizer type!
                msg = prompt_builder.add_turn(turn["from"], turn["value"])

                # Llama Tokenizer (Fast) adds extra character if a string ends in whitespace --> strip if non-empty!
                if isinstance(self.tokenizer, LlamaTokenizerFast):
                    msg = msg.rstrip()
                else:
                    raise ValueError(f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!")

                # Tokenize Input IDs
                turn_input_ids = self.tokenizer(msg, add_special_tokens=turn_idx == 0).input_ids

                # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
                turn_labels = (
                    [IGNORE_INDEX for _ in range(len(turn_input_ids))] if (turn_idx % 2) == 0 else list(turn_input_ids)
                )

                # Add to Trackers
                input_ids.extend(turn_input_ids)
                labels.extend(turn_labels)

        # Tensorize =>> Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches after)
        #   - IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        # Handle Truncation (if necessary)
        input_ids, labels = input_ids[: self.tokenizer.model_max_length], labels[: self.tokenizer.model_max_length]

    
        return dict(input_ids=input_ids, labels=labels)



    def __len__(self) -> int:
        return len(self.examples)

        

class FinetuneDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        instruct_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        super().__init__()
        self.instruct_json, self.image_dir = instruct_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset_type = "finetune"

        # Load Instruct JSON
        with open(self.instruct_json, "r") as f:
            self.examples = json.load(f)

    # === Unimodal + Multimodal Handling ===
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Unlike the *align* stage handling, for the *finetune* stage, we actually need to handle multiple "turns" of
        dialog grounded in a single image.

        To do this, we leverage the `prompt_builder_fn` which instantiates a PromptBuilder object. By calling the
        methods for adding turns and getting a prompt, we ensure proper formatting and consistency for each example.

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        conversation = self.examples[idx]["conversations"]
        # conversation_cp = copy.deepcopy(conversation)
        # conversation = None

        # Create Prompt Builder --> add each message sequentially
        prompt_builder, input_ids, labels = self.prompt_builder_fn(model_family="prismatic"), [], []
        for turn_idx, turn in enumerate(conversation):
            # Get "effective" string added to prompt --> handle whitespace for tokenizer type!
            msg = prompt_builder.add_turn(turn["from"], turn["value"])

            # Llama Tokenizer (Fast) adds extra character if a string ends in whitespace --> strip if non-empty!
            if isinstance(self.tokenizer, LlamaTokenizerFast):
                msg = msg.rstrip()
            else:
                raise ValueError(f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!")

            # Tokenize Input IDs
            turn_input_ids = self.tokenizer(msg, add_special_tokens=turn_idx == 0).input_ids

            # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
            turn_labels = (
                [IGNORE_INDEX for _ in range(len(turn_input_ids))] if (turn_idx % 2) == 0 else list(turn_input_ids)
            )

            # Add to Trackers
            input_ids.extend(turn_input_ids)
            labels.extend(turn_labels)

        # Tensorize =>> Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches after)
        #   - IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        # Handle Truncation (if necessary)
        input_ids, labels = input_ids[: self.tokenizer.model_max_length], labels[: self.tokenizer.model_max_length]

        # === Handle "unimodal" (language-only) vs. "multimodal" ===
        if "image" in self.examples[idx]:
            image_path = Path(self.examples[idx]["image"])

            # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
            labels[0] = IGNORE_INDEX

            # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
            pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert("RGB"))
            
            # pixel_values_cp = copy.deepcopy(pixel_values)
            # pixel_values = None
            # input_ids_cp = copy.deepcopy(input_ids)
            # input_ids = None
            # labels_cp = copy.deepcopy(labels)
            # labels = None

            return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

        else:
            # No image --> return `pixel_values` = None; Collator will do the smart batch handling for us!
            # input_ids_cp = copy.deepcopy(input_ids)
            # input_ids = None
            # labels_cp = copy.deepcopy(labels)
            # labels = None
            return dict(pixel_values=None, input_ids=input_ids, labels=labels)

    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example
            n_words = sum([len(turn["value"].split()) for turn in example["conversations"]])
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)
