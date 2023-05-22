import logging
from functools import partial

import evaluate
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning_hivemind.strategy import HivemindStrategy
from toolz import compose_left
from transformers import GenerationConfig, PreTrainedModel

from llms import CONFIGS_PATH, DATASETS_PATH, METRICS_PATH
from llms.training.datamodule import Seq2SeqDataModule
from llms.training.factory import get_model, get_peft_config
from llms.training.preprocessing import preprocess_batch
from llms.training.transforms import TrainingTransform
from llms.training.wrapper import Seq2SeqWrapper
from llms.utils.config import load_config
from llms.utils.dvc import maybe_get_dvc

logging.basicConfig(level=logging.INFO)

config = load_config(maybe_get_dvc(CONFIGS_PATH / "model_config.yaml"))

peft_config = get_peft_config(config)
model, tokenizer = get_model(
    model_name=config["model"]["model_name"],
    load_in_8bit=config["model"]["load_in_8bit"],
    peft_config=peft_config,
)


def configure_optimizers_func(model: PreTrainedModel):
    optimizer = torch.optim.AdamW(
        params=[param for param in model.parameters() if param.requires_grad],
        lr=config["optimizer"]["lr"],
    )
    return optimizer


metrics = [
    evaluate.load(str(METRICS_PATH / f"{metric_name}.py"))
    for metric_name in config["metrics"]["metrics"]
]
wrapper = Seq2SeqWrapper(
    model=model,
    tokenizer=tokenizer,
    configure_optimizers_func=configure_optimizers_func,
    metrics=metrics,
    generation_config=GenerationConfig(
        max_new_tokens=config["datamodule"]["max_target_length"],
        temperature=config["generation"]["temperature"],
    ),
)

dataset_path = DATASETS_PATH / f"{config['datamodule']['dataset_name']}.py"
transforms = [
    TrainingTransform(),
    partial(
        preprocess_batch,
        tokenizer=tokenizer,
        max_context_length=config["datamodule"]["max_context_length"],
        max_target_length=config["datamodule"]["max_target_length"],
    ),
]
datamodule = Seq2SeqDataModule(
    dataset_path=dataset_path,
    tokenizer=tokenizer,
    batch_size=config["datamodule"]["batch_size"],
    num_workers=config["datamodule"]["num_workers"],
    transform_func=compose_left(*transforms),
)

logger = MLFlowLogger(
    experiment_name="llms",
)
logger.log_hyperparams(config)
trainer = Trainer(
    max_epochs=config["trainer"]["max_epochs"],
    accumulate_grad_batches=config["trainer"]["accumulate_grad_batches"],
    accelerator=config["trainer"]["accelerator"],
    precision=config["trainer"]["precision"],
    logger=logger,
    callbacks=[
        ModelCheckpoint(
            monitor=f'validation/{config["metrics"]["monitor"]["name"]}',
            mode=config["metrics"]["monitor"]["mode"],
        ),
    ],
    limit_val_batches=config["trainer"]["limit_val_batches"],
    check_val_every_n_epoch=config["trainer"]["check_val_every_n_epoch"],
    strategy=HivemindStrategy(target_batch_size=4),
)

trainer.fit(
    model=wrapper,
    datamodule=datamodule,
)
