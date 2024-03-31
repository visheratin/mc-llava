import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from conversation import Conversation


class TextvqaDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        query_tokenizer,
    ) -> None:
        super().__init__()
        self.dataset = load_dataset("lmms-lab/textvqa", split="train", num_proc=6)
        self.tokenizer = tokenizer
        self.query_tokenizer = query_tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item["question"]
        query_input_ids = self.query_tokenizer(
            question, return_tensors="pt"
        ).input_ids.squeeze(0)
        answer = item["answers"][0]
        question = f"<image>\n{question}"
        conv = Conversation([question, answer])
        _, input_ids, labels = conv.get_prompt(self.tokenizer)

        image = item["image"].convert("RGB")
        return (
            input_ids,
            labels,
            image,
            query_input_ids,
        )
