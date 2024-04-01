from dataclasses import dataclass
from typing import Union

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from dataset import TrainingDataset
from processing_mc_llava import MultiCropImageProcessor


@dataclass
class DataCollator:
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    processor: MultiCropImageProcessor

    def __call__(self, batch):
        input_ids = []
        labels = []
        images = []
        for item in batch:
            input_ids.append(item[0])
            labels.append(item[1])
            images.append(item[2])
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        attention_mask = attention_mask[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        pixel_values, coords = self.processor(images)
        return input_ids, attention_mask, labels, pixel_values, coords


class TrainingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        data_dir: str = "./data",
        max_model_length: int = 0,
        crops_limit: int = 64,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.images_dir = data_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            "visheratin/MC-LLaVA-3b", trust_remote_code=True
        )
        if max_model_length > 0:
            self.tokenizer.model_max_length = max_model_length
        self.processor = MultiCropImageProcessor("visheratin/MC-LLaVA-3b", crops_limit)
        self.dataset = TrainingDataset(data_dir, self.tokenizer)

    def setup(self, stage=None):
        pass

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=0,
            collate_fn=DataCollator(self.tokenizer, self.processor),
            persistent_workers=True,
            pin_memory=True,
        )
