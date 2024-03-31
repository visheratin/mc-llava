import random

import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import Dataset

from conversation import Conversation


class UberDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        query_tokenizer,
    ) -> None:
        super().__init__()
        dataset = load_dataset("visheratin/uber_text_qa", num_proc=6)
        self.dataset = concatenate_datasets([dataset["train"], dataset["val"]])
        self.tokenizer = tokenizer
        self.query_tokenizer = query_tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if len(item["questions"]) == 0:
            return self.__getitem__(idx + 1)
        question_idx = random.randrange(0, len(item["questions"]))
        question = item["questions"][question_idx]
        answer = item["answers"][question_idx]
        query_input_ids = self.query_tokenizer(
            item.messages[0], return_tensors="pt"
        ).input_ids.squeeze(0)
        question = f"<image>\n{question}"
        conv = Conversation([question, answer])
        _, input_ids, labels = conv.get_prompt(self.tokenizer)

        image = item["image"]
        return (
            input_ids,
            labels,
            image,
            query_input_ids,
        )
