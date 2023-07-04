import os
from pathlib import Path

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from llms.training.wrapper import Seq2SeqWrapper


class TransformersModelCheckpoint(ModelCheckpoint):
    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        model_wrapper = trainer.model
        if not isinstance(model_wrapper, Seq2SeqWrapper):
            raise ValueError(
                f"Model wrapper should be of type {Seq2SeqWrapper}, but is {type(model_wrapper)}"
            )

        model_path = Path(filepath).parent / "model"
        model_path.mkdir(parents=True, exist_ok=True)
        model_wrapper.model_wrapper.save_pretrained(str(model_path))
