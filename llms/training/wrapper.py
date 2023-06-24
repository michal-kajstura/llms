from copy import deepcopy
from typing import Any, Sequence

import torch
from evaluate import EvaluationModule
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    SchedulerType,
    get_scheduler,
)

from llms.configs.training import (
    AdamWConfig,
    LinearSchedulerWithWarmupConfig,
    OptimizerConfig,
    SchedulerConfig,
)


class Seq2SeqWrapper(LightningModule):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        metrics: Sequence[EvaluationModule],
        generation_config: GenerationConfig,
        optimizer_config: OptimizerConfig,
        scheduler_config: SchedulerConfig,
    ):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._metrics = {epoch_type: deepcopy(metrics) for epoch_type in ["validation", "test"]}
        self._generation_config = generation_config
        self._optimizer_config = optimizer_config
        self._scheduler_config = scheduler_config

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> STEP_OUTPUT:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx) -> STEP_OUTPUT:
        self._evaluation_step(batch, "validation")
        return self._step(batch, "validation")

    def test_step(self, batch: dict[str, Tensor], batch_idx) -> STEP_OUTPUT:
        self._evaluation_step(batch, "test")
        return self._step(batch, "test")

    def _step(self, batch: dict[str, Tensor], epoch_type: str):
        outputs = self._model(**batch)
        self.log(f"{epoch_type}/loss", outputs.loss)
        return outputs.loss

    def _evaluation_step(self, batch: dict[str, Tensor], epoch_type: str):
        generated_tokens = self._model.generate(
            input_ids=batch["input_ids"],
            generation_config=self._generation_config,
        )
        decoded_text = self._tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        labels = batch["labels"].clone()
        labels[labels == -100] = self._tokenizer.pad_token_id
        decoded_reference = self._tokenizer.batch_decode(labels, skip_special_tokens=True)

        for metric in self._metrics[epoch_type]:
            metric.add_batch(predictions=decoded_text, references=decoded_reference)

    def on_validation_epoch_end(self) -> None:
        self._on_evaluation_epoch_end("validation")

    def on_test_epoch_end(self) -> None:
        self._on_evaluation_epoch_end("test")

    def _on_evaluation_epoch_end(self, epoch_type: str) -> None:
        for metric in self._metrics[epoch_type]:
            output = metric.compute()
            self.log_dict({f"{epoch_type}/{name}": value for name, value in output.items()})

    def configure_optimizers(self) -> Any:
        match self._optimizer_config:
            case AdamWConfig(lr=lr, weight_decay=weight_decay, eps=eps, betas=betas):
                optimizer = torch.optim.AdamW(
                    params=[param for param in self._model.parameters() if param.requires_grad],
                    lr=lr,
                    weight_decay=weight_decay,
                    eps=eps,
                    betas=betas,
                )
            case _:
                raise NotImplementedError(self._optimizer_config)

        total_devices = self.trainer.num_devices * self.trainer.num_nodes
        train_batches = len(self.trainer.datamodule.train_dataloader()) // total_devices
        num_training_steps = (
            self.trainer.max_epochs * train_batches
        ) // self.trainer.accumulate_grad_batches
        match self._scheduler_config:
            case LinearSchedulerWithWarmupConfig(num_warmup_steps=num_warmup_steps):
                scheduler = get_scheduler(
                    name=SchedulerType.LINEAR,
                    optimizer=optimizer,
                    num_training_steps=num_training_steps,
                    num_warmup_steps=num_warmup_steps,
                )
            case _:
                raise NotImplementedError(self._scheduler_config)

        return {
            "optimizer": optimizer,
            "lr_scheduler_config": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
