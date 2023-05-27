import logging

import evaluate
import lightning
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from toolz import compose_left
from transformers import GenerationConfig, PreTrainedModel

from llms import DATASETS_PATH, METRICS_PATH
from llms.configs.training import TrainingConfig
from llms.training.datamodule import Seq2SeqDataModule
from llms.training.factory import get_model, get_peft_config
from llms.training.preprocessing import PreprocessBatch, TransformFields
from llms.training.wrapper import Seq2SeqWrapper

logging.basicConfig(level=logging.INFO)

config = TrainingConfig()
lightning.seed_everything(config.seed)

peft_config = get_peft_config(config)
model, tokenizer = get_model(
    model_name=config.model.model_name,
    load_in_8bit=config.model.load_in_8bit,
    load_in_4bit=config.model.load_in_4bit,
    peft_config=peft_config,
)


def configure_optimizers_func(model: PreTrainedModel):
    optimizer = torch.optim.AdamW(
        params=[param for param in model.parameters() if param.requires_grad],
        lr=config.optimizer.lr,
    )
    return optimizer


metrics = [
    evaluate.load(
        str(METRICS_PATH / f"{metric_name}.py"),
        **getattr(config.metrics, "metric_name", {}),
    )
    for metric_name in config.metrics.used_metrics
]
wrapper = Seq2SeqWrapper(
    model=model,
    tokenizer=tokenizer,
    configure_optimizers_func=configure_optimizers_func,
    metrics=metrics,
    generation_config=GenerationConfig(
        max_new_tokens=config.preprocessing.max_target_length,
        temperature=config.generation.temperature,
    ),
)

dataset_path = DATASETS_PATH / f"{config.datamodule.dataset_name}.py"
eval_transforms = [
    PreprocessBatch(
        tokenizer=tokenizer,
        max_context_length=config.preprocessing.max_context_length,
        max_target_length=config.preprocessing.max_target_length,
        answer_delimiter=config.preprocessing.answer_delimiter,
        line_delimiter=config.preprocessing.line_delimiter,
        tab_delimiter=config.preprocessing.tab_delimiter,
    ),
]
training_transforms = [
    TransformFields(
        min_num_fields=config.preprocessing.min_num_fields,
        max_num_fields=config.preprocessing.max_num_fields,
    ),
    *eval_transforms,
]
datamodule = Seq2SeqDataModule(
    dataset_path=dataset_path,
    tokenizer=tokenizer,
    batch_size=config.datamodule.batch_size,
    num_workers=config.datamodule.num_workers,
    training_transform_func=compose_left(*training_transforms),
    eval_transform_func=compose_left(*eval_transforms),
)

logger = MLFlowLogger(
    experiment_name="llms",
)
logger.log_hyperparams(config)
trainer = Trainer(
    max_epochs=config.trainer.max_epochs,
    accumulate_grad_batches=config.trainer.accumulate_grad_batches,
    accelerator=config.trainer.accelerator,
    precision=config.trainer.precision,
    logger=logger,
    callbacks=[
        ModelCheckpoint(
            monitor=f"validation/{config.metrics.monitor.name}",
            mode=config.metrics.monitor.mode,
        ),
    ],
    limit_val_batches=config.trainer.limit_val_batches,
    check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
)

trainer.validate(
    model=wrapper,
    datamodule=datamodule,
)
trainer.fit(
    model=wrapper,
    datamodule=datamodule,
)
