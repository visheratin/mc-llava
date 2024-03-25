import json
import os
import random
from dataclasses import dataclass
from io import BytesIO
from typing import List

import requests
import torch
from PIL import Image
from torch.utils.data import Dataset

from conversation import Conversation


@dataclass
class DataItem:
    id: str
    messages: List[str]


class JsonDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        images_dir: str,
        tokenizer,
        max_size: int,
    ) -> None:
        super().__init__()
        with open(file_path, "r") as f:
            data = json.load(f)
        self.data: List[DataItem] = []
        for item in data:
            id = item["id"]
            messages = []
            for conv_item in item["conversations"]:
                messages.append(conv_item["value"])
            self.data.append(DataItem(id, messages))
        if max_size > 0:
            self.data = random.choices(self.data, k=max_size)
        self.tokenizer = tokenizer
        self.images_dir = images_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        conv = Conversation(item.messages)
        _, input_ids, labels = conv.get_prompt(self.tokenizer)
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)

        image = None
        try:
            image = self.open_image(item.id)
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
