import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from conversation import Conversation


class TextvqaDataset(Dataset):
    def __init__(
        self,
        tokenizer,
    ) -> None:
        super().__init__()
        self.dataset = load_dataset("lmms-lab/textvqa", split="train", num_proc=6)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item["question"]
        answer = item["answers"][0]
        question = f"<image>\n{question}"
        conv = Conversation([question, answer])
        _, input_ids, labels = conv.get_prompt(self.tokenizer)
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)

        image = item["image"].convert("RGB")
        return (
            input_ids,
            attention_mask,
            labels,
            image,
        )
