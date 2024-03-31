import os
import random
from io import BytesIO

import requests
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset

from conversation import Conversation


class SvitDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        tokenizer,
        query_tokenizer,
        max_size: int = 0,
    ) -> None:
        super().__init__()
        self.images_dir = images_dir
        os.makedirs(images_dir, exist_ok=True)
        dataset = load_dataset("visheratin/SVIT")
        self.dataset = dataset["train"]
        if max_size > 0:
            self.dataset = self.dataset.shuffle()
            self.dataset = self.dataset.select(list(range(max_size)))
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
            question, return_tensors="pt"
        ).input_ids.squeeze(0)
        question = f"<image>\n{question}"
        conv = Conversation([question, answer])
        _, input_ids, labels = conv.get_prompt(self.tokenizer)
        image = None
        try:
            image = self.open_image(item["id"], item["url"])
        except:
            return self.__getitem__(idx + 1)
        return (
            input_ids,
            labels,
            image,
            query_input_ids,
        )

    def open_image(self, id: str, url: str):
        file_path = os.path.join(self.images_dir, f"{id}.jpg")
        if os.path.exists(file_path):
            return Image.open(file_path).convert("RGB")
        else:
            response = requests.get(url, timeout=10)
            image = Image.open(BytesIO(response.content))
            image = image.convert("RGB")
            image.save(file_path)
            return image
