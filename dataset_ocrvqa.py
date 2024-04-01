import random

import torch
from datasets import VerificationMode, load_dataset
from torch.utils.data import Dataset

from conversation import Conversation


class OcrvqaDataset(Dataset):
    def __init__(
        self,
        tokenizer,
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
            verification_mode=VerificationMode.NO_CHECKS,
        )
        self.tokenizer = tokenizer

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

        image = item["image"].convert("RGB")
        return (
            input_ids,
            labels,
            image,
        )
