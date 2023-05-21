from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizer

from llms.training.preprocessing import preprocess_data
from llms.training.transform import TrainingTransform


class Seq2SeqDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        num_workers: int = -1,
        max_context_length: int = 512,
        max_target_length: int = 64,
    ) -> None:
        super().__init__()
        self._dataset_path = dataset_path
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._max_context_length = max_context_length
        self._max_target_length = max_target_length

    def prepare_data(self):
        dataset = load_dataset(str(self._dataset_path))
        self._tokenized_dataset = preprocess_data(
            dataset=dataset,
            tokenizer=self._tokenizer,
            max_context_length=self._max_context_length,
            max_target_length=self._max_target_length,
            transform=TrainingTransform(),
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._get_dataloader(split="train", shuffle=True)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return self._get_dataloader(split="validation", shuffle=False)

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return self._get_dataloader(split="test", shuffle=False)

    def _get_dataloader(self, split: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            self._tokenized_dataset[split],
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
