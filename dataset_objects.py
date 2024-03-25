import os
import random
from io import BytesIO

import requests
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset

from conversation import Conversation


class ObjectsDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        tokenizer,
        max_size: int,
    ) -> None:
        super().__init__()
        self.images_dir = images_dir
        os.makedirs(images_dir, exist_ok=True)
        dataset = load_dataset("visheratin/object_questions")
        self.dataset = dataset["train"]
        if max_size > 0:
            self.dataset = self.dataset.shuffle()
            self.dataset = self.dataset.select(list(range(max_size)))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if len(item["questions"]) == 0:
            return self.__getitem__(idx + 1)
        question_idx = random.randrange(0, len(item["questions"]))
        question = item["questions"][question_idx]
        answer = item["answers"][question_idx]
        question = f"<image>\n{question}"
        conv = Conversation([question, answer])
        _, input_ids, labels = conv.get_prompt(self.tokenizer)
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)

        image = None
        try:
            image = self.open_image(item["id"])
        except:
            return self.__getitem__(idx + 1)
        return (
            input_ids,
            attention_mask,
            labels,
            image,
        )

    def open_image(self, id: str):
        file_path = os.path.join(self.images_dir, f"{id}.jpg")
        if os.path.exists(file_path):
            return Image.open(file_path).convert("RGB")
        else:
            url = f"https://nllb-data.com/{id}.jpg"
            response = requests.get(url, timeout=10)
            image = Image.open(BytesIO(response.content))
            image = image.convert("RGB")
            image.save(file_path)
            return image
