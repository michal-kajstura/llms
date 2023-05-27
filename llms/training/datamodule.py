from collections.abc import Callable, Mapping
from typing import Any

from datasets import load_dataset
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from toolz import identity
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizer


class Seq2SeqDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        num_workers: int = -1,
        training_transform_func: Callable[
            [Mapping[str, Any]], Mapping[str, Any]
        ] = identity,
        eval_transform_func: Callable[
            [Mapping[str, Any]], Mapping[str, Any]
        ] = identity,
    ) -> None:
        super().__init__()
        self._dataset_path = dataset_path
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._training_transform_func = training_transform_func
        self._eval_transform_func = eval_transform_func

    def prepare_data(self):
        self._dataset = load_dataset(str(self._dataset_path))
        for split, dataset in self._dataset.items():
            transform_func = (
                self._training_transform_func
                if split == "train"
                else self._eval_transform_func
            )
            dataset.set_transform(transform_func)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._get_dataloader(split="train", shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_dataloader(split="validation", shuffle=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_dataloader(split="test", shuffle=False)

    def _get_dataloader(self, split: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            self._dataset[split],
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=shuffle,
            collate_fn=DataCollatorForSeq2Seq(
                tokenizer=self._tokenizer,
                padding="longest",
                pad_to_multiple_of=8,
                return_tensors="pt",
            ),
        )
