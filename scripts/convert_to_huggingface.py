import logging

import evaluate
import lightning
import torch
from transformers import GenerationConfig

from llms import METRICS_PATH
from llms.configs.training import TrainingConfig
from llms.training.factory import get_model
from llms.training.wrapper import Seq2SeqWrapper

logging.basicConfig(level=logging.INFO)

config = TrainingConfig()
config.model.modify_config(config)

lightning.seed_everything(config.seed)

model_wrapper = get_model(config)

metrics = [
    evaluate.load(
        str(METRICS_PATH / f"{metric_name}.py"),
        **getattr(config.metrics, "metric_name", {}),
    )
    for metric_name in config.metrics.used_metrics
]

checkpoint_path = "/home/mkajstura/projects/llms/647053360652379201/6d5d3520be214efbb69202abd1e3502a/checkpoints/epoch=7-step=3760.ckpt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
print(checkpoint['state_dict'])

print(checkpoint.keys())
training_wrapper = Seq2SeqWrapper.load_from_checkpoint(
    checkpoint_path,
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
)

print(training_wrapper)
