import random

import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import Dataset

from conversation import Conversation
from processing_mc_llava import MultiCropImageProcessor


class UberDataset(Dataset):
    def __init__(
        self, tokenizer, processor: MultiCropImageProcessor, crops_limit: int
    ) -> None:
        super().__init__()
        dataset = load_dataset("visheratin/uber_text_qa", num_proc=6)
        self.dataset = concatenate_datasets([dataset["train"], dataset["val"]])
        self.tokenizer = tokenizer
        self.processor = processor
        self.crop_limit = crops_limit

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if len(item["questions"]) == 0:
            return self.__getitem__(idx+1)
        question_idx = random.randrange(0, len(item["questions"]))
        question = item["questions"][question_idx]
        answer = item["answers"][question_idx]
        question = f"<image>\n{question}"
        conv = Conversation([question, answer])
        _, input_ids, labels = conv.get_prompt(self.tokenizer)
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)

        image = item["image"]
        max_crops = random.randint(0, self.crop_limit)
        image_res = self.processor([image], max_crops)
        return (
            input_ids,
            attention_mask,
            labels,
            image_res["pixel_values"],
            image_res["coords"],
        )
