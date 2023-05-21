from collections.abc import Callable
from typing import Any, Sequence

from evaluate import EvaluationModule
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer


class Seq2SeqWrapper(LightningModule):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        configure_optimizers_func: Callable[[PreTrainedModel], Any],
        metrics: Sequence[EvaluationModule],
        generation_config: GenerationConfig,
    ):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._configure_optimizers_func = configure_optimizers_func
        self._metrics = {epoch_type: metrics for epoch_type in ["validation", "test"]}
        self._generation_config = generation_config

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
            inputs=batch["input_ids"],
            generation_config=self._generation_config,
        )
        decoded_text = self._tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        labels = batch["labels"].clone()
        labels[labels == -100] = self._tokenizer.pad_token_id
        decoded_reference = self._tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

        for metric in self._metrics[epoch_type]:
            metric.add_batch(predictions=decoded_text, references=decoded_reference)

    def on_validation_epoch_end(self) -> None:
        self._on_evaluation_epoch_end("validation")

    def on_test_epoch_end(self) -> None:
        self._on_evaluation_epoch_end("test")

    def _on_evaluation_epoch_end(self, epoch_type: str) -> None:
        for metric in self._metrics[epoch_type]:
            output = metric.compute()
            self.log_dict(
                {f"{epoch_type}/{name}": value for name, value in output.items()}
            )

    def configure_optimizers(self) -> Any:
        return self._configure_optimizers_func(self._model)
