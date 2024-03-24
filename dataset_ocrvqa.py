import random

import torch
from datasets import load_dataset, VerificationMode
from torch.utils.data import Dataset

from conversation import Conversation
from processing_mc_llava import MultiCropImageProcessor


class OcrvqaDataset(Dataset):
    def __init__(
        self, tokenizer, processor: MultiCropImageProcessor, crops_limit: int
    ) -> None:
        super().__init__()
        self.dataset = load_dataset(
            "howard-hou/OCR-VQA",
            split="train",
            data_files=[
                "data/train-00000-of-00016-7de80f046ddc3f89.parquet",
                "data/train-00001-of-00016-26903f357666ec36.parquet",
            ],
            num_proc=2,
            verification_mode=VerificationMode.NO_CHECKS
        )
        self.tokenizer = tokenizer
        self.processor = processor
        self.crop_limit = crops_limit

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question_idx = random.randrange(0, len(item["questions"]))
        question = item["questions"][question_idx]
        answer = item["answers"][question_idx]
        question = f"<image>\n{question}"
        conv = Conversation([question, answer])
        _, input_ids, labels = conv.get_prompt(self.tokenizer)
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)

        image = item["image"].convert("RGB")
        max_crops = random.randint(0, self.crop_limit)
        image_res = self.processor([image], max_crops)
        return (
            input_ids,
            attention_mask,
            labels,
            image_res["pixel_values"],
            image_res["coords"],
        )
