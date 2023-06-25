import logging
from pathlib import Path

import evaluate
import lightning
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from toolz import compose_left
from transformers import GenerationConfig, PreTrainedModel

from llms import METRICS_PATH, datasets as datasets_module
from llms.configs.training import TrainingConfig
from llms.training.datamodule import Seq2SeqDataModule
from llms.training.factory import get_model
from llms.training.preprocessing import PreprocessBatch, TransformFields
from llms.training.wrapper import Seq2SeqWrapper

logging.basicConfig(level=logging.INFO)

config = TrainingConfig()
config.model.modify_config(config)
print(config)


lightning.seed_everything(config.seed)

model_wrapper = get_model(config)


def configure_optimizers_func(model: PreTrainedModel):
    pass


metrics = [
    evaluate.load(
        str(METRICS_PATH / f"{metric_name}.py"),
        **getattr(config.metrics, "metric_name", {}),
    )
    for metric_name in config.metrics.used_metrics
]
wrapper = Seq2SeqWrapper(
    model_wrapper=model_wrapper,
    metrics=metrics,
    generation_config=GenerationConfig(
        max_new_tokens=config.data.max_target_length,
        temperature=config.model.temperature,
        eos_token_id=model_wrapper.tokenizer.eos_token_id,
        pad_token_id=model_wrapper.tokenizer.pad_token_id,
    ),
    optimizer_config=config.optimizer,
    scheduler_config=config.scheduler,
    to_save=config.dict(),
)

print('\n\n\n')
print('Prompt:')
print(config.data.prompt_template)
print('\n\n\n')
eval_transforms = [
    PreprocessBatch(
        tokenizer=model_wrapper.tokenizer,
        max_context_length=config.data.max_context_length,
        max_target_length=config.data.max_target_length,
        answer_delimiter=config.data.answer_delimiter,
        line_delimiter=config.data.line_delimiter,
        tab_delimiter=config.data.tab_delimiter,
        normalize_text=config.data.normalize_text,
    ),
]
training_transforms = [
    TransformFields(
        min_num_fields=config.data.min_num_fields,
        max_num_fields=config.data.max_num_fields,
    ),
    *eval_transforms,
]
dataset_path = Path(datasets_module.__file__).parent / f"{config.data.dataset_name}.py"
datamodule = Seq2SeqDataModule(
    dataset_path=str(dataset_path),
    data_collator=model_wrapper.data_collator,
    batch_size=config.trainer.batch_size,
    num_workers=config.trainer.num_workers,
    training_transform_func=compose_left(*training_transforms),
    eval_transform_func=compose_left(*eval_transforms),
)

logger = MLFlowLogger(
    experiment_name="llms",
)
logger.log_hyperparams(config.dict())
trainer = Trainer(
    max_epochs=config.trainer.max_epochs,
    accumulate_grad_batches=config.trainer.accumulate_grad_batches,
    accelerator=config.trainer.accelerator,
    precision=config.trainer.precision,
    logger=logger,
    callbacks=[
        ModelCheckpoint(
            monitor=f"validation/{config.metrics.main_metric}",
            mode=config.metrics.mode,
        ),
        LearningRateMonitor(
            logging_interval="step",
        ),
    ],
    check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
)

# trainer.validate(
#     model=wrapper,
#     datamodule=datamodule,
# )
trainer.fit(
    model=wrapper,
    datamodule=datamodule,
)
