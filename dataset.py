import os

from torch.utils.data import Dataset

from dataset_json import JsonDataset
from dataset_landmarks import LandmarksDataset
from dataset_objects import ObjectsDataset
from dataset_ocrvqa import OcrvqaDataset
from dataset_textvqa import TextvqaDataset
from dataset_svit import SvitDataset
from dataset_uber import UberDataset


class TrainingDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        tokenizer,
    ):
        super(TrainingDataset, self).__init__()
        self.tokenizer = tokenizer

        landmarks_root = os.path.join(data_root, "landmarks")
        landmarks_dataset = LandmarksDataset(landmarks_root, self.tokenizer)
        uber_dataset = UberDataset(self.tokenizer)
        objects_dataset = ObjectsDataset(
            os.path.join(data_root, "laion"),
            self.tokenizer,
            50000,
        )
        ocrvqa_dataset = OcrvqaDataset(
            self.tokenizer,
        )
        textvqa_dataset = TextvqaDataset(
            self.tokenizer,
        )
        captions_dataset = JsonDataset(
            os.path.join(data_root, "captions.json"),
            os.path.join(data_root, "laion"),
            self.tokenizer,
            150000,
        )
        questions_dataset = JsonDataset(
            os.path.join(data_root, "questions.json"),
            os.path.join(data_root, "laion"),
            self.tokenizer,
            80000,
        )
        svit_dataset = SvitDataset(
            os.path.join(data_root, "svit"),
            self.tokenizer,
        )
        self.datasets = [
            # warmup + stable
            svit_dataset, # 108,000
            objects_dataset, #50,000
            captions_dataset, #150,000
            # annealing + low lr
            ocrvqa_dataset, # 25,000
            questions_dataset, #80,000
            # low lr
            landmarks_dataset, # 35,000
            uber_dataset, # 10,000
            textvqa_dataset, # 45,000
        ]
        self.dataset_lengths = [len(d) for d in self.datasets]
        self.total_length = sum(self.dataset_lengths)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self._index_in_datasets(idx)
        return self.datasets[dataset_idx][sample_idx]

    def _index_in_datasets(self, idx):
        if idx < 0 or idx >= self.total_length:
            raise IndexError("Index out of range")
        for i, length in enumerate(self.dataset_lengths):
            if idx < length:
                return i, idx
            idx -= length
        raise IndexError("Index calculation error")
