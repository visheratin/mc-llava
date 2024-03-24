import lightning.pytorch as pl
from torch.utils.data import DataLoader

from dataset import TrainingDataset
import torch

def collate_fn(batch):
    input_ids = []
    attention_mask = []
    labels = []
    images = []
    coords = []
    for item in batch:
        input_ids.append(item[0])
        attention_mask.append(item[1])
        labels.append(item[2])
        images.append(item[3][0])
        coords.append(item[4][0])
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    labels = torch.stack(labels)
    images = torch.nested.nested_tensor(images)
    coords = torch.nested.nested_tensor(coords)
    return input_ids, attention_mask, labels, images, coords


class TrainingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        data_dir: str = "./data",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.images_dir = data_dir

        self.dataset = TrainingDataset(data_dir, crops_limit=1)

    def setup(self, stage=None):
        pass

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=12,
            collate_fn=collate_fn,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )
