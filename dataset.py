import os

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from dataset_json import JsonDataset
from dataset_landmarks import LandmarksDataset
from dataset_objects import ObjectsDataset
from dataset_ocrvqa import OcrvqaDataset
from dataset_svit import SvitDataset
from dataset_textvqa import TextvqaDataset
from dataset_uber import UberDataset
from processing_mc_llava import MultiCropImageProcessor


class TrainingDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        crops_limit: int = 8,
    ):
        super(TrainingDataset, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "visheratin/MC-LLaVA-3b", trust_remote_code=True
        )
        self.query_tokenizer = AutoTokenizer.from_pretrained(
            "google/siglip-so400m-patch14-384"
        )
        self.processor = MultiCropImageProcessor("visheratin/MC-LLaVA-3b", crops_limit)

        landmarks_root = os.path.join(data_root, "landmarks")
        landmarks_dataset = LandmarksDataset(
            landmarks_root, self.tokenizer, self.query_tokenizer
        )
        uber_dataset = UberDataset(self.tokenizer, self.query_tokenizer)
        objects_dataset = ObjectsDataset(
            os.path.join(data_root, "laion"),
            self.tokenizer,
            self.query_tokenizer,
            35000,
        )
        ocrvqa_dataset = OcrvqaDataset(
            self.tokenizer,
            self.query_tokenizer,
        )
        textvqa_dataset = TextvqaDataset(
            self.tokenizer,
            self.query_tokenizer,
        )
        captions_dataset = JsonDataset(
            os.path.join(data_root, "captions.json"),
            os.path.join(data_root, "laion"),
            self.tokenizer,
            self.query_tokenizer,
            100000,
        )
        questions_dataset = JsonDataset(
            os.path.join(data_root, "questions.json"),
            os.path.join(data_root, "laion"),
            self.tokenizer,
            self.query_tokenizer,
            50000,
        )
        svit_dataset = SvitDataset(
            os.path.join(data_root, "svit"),
            self.tokenizer,
            self.query_tokenizer,
        )
        self.datasets = [
            landmarks_dataset,
            uber_dataset,
            objects_dataset,
            captions_dataset,
            questions_dataset,
            ocrvqa_dataset,
            textvqa_dataset,
            svit_dataset,
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
