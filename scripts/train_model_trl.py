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
from llms.training.collator import DataCollatorWithPrompt
from llms.training.datamodule import Seq2SeqDataModule
from llms.training.factory import get_model
from llms.training.preprocessing import PreprocessBatch, TransformFields
from llms.training.wrapper import Seq2SeqWrapper

logging.basicConfig(level=logging.INFO)

config = TrainingConfig()
config.model.modify_config(config)

lightning.seed_everything(config.seed)

# model_wrapper = get_model(config)
#

# def configure_optimizers_func(model: PreTrainedModel):
#     pass
#
#
# metrics = [
#     evaluate.load(
#         str(METRICS_PATH / f"{metric_name}.py"),
#         **getattr(config.metrics, "metric_name", {}),
#     )
#     for metric_name in config.metrics.used_metrics
# ]
# wrapper = Seq2SeqWrapper(
#     model_wrapper=model_wrapper,
#     metrics=metrics,
#     generation_config=GenerationConfig(
#         max_new_tokens=config.data.max_target_length,
#         temperature=config.model.temperature,
#         eos_token_id=model_wrapper.tokenizer.eos_token_id,
#         pad_token_id=model_wrapper.tokenizer.pad_token_id,
#     ),
#     optimizer_config=config.optimizer,
#     scheduler_config=config.scheduler,
#     to_save=config.dict(),
# )


import torch, einops
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments
)
from peft.tuners.lora import LoraLayer

from trl import SFTTrainer

def create_and_prepare_model():
    compute_dtype = getattr(torch, "float16")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "tiiuae/falcon-7b", quantization_config=bnb_config, device_map={"": 0}, trust_remote_code=True
    )

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "query_key_value"
        ],
    )

    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer


training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=10,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=10000,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
)

model, peft_config, tokenizer = create_and_prepare_model()
eval_transforms = [
    # PreprocessBatch(
    #     tokenizer=tokenizer,
    #     prompt_template=config.data.prompt_template,
    #     max_context_length=config.data.max_context_length,
    #     max_target_length=config.data.max_target_length,
    #     answer_delimiter=config.data.answer_delimiter,
    #     line_delimiter=config.data.line_delimiter,
    #     tab_delimiter=config.data.tab_delimiter,
    #     normalize_text=config.data.normalize_text,
    # ),
]
training_transforms = [
    # TransformFields(
    #     min_num_fields=config.data.min_num_fields,
    #     max_num_fields=config.data.max_num_fields,
    # ),
    # *eval_transforms,
]
dataset_path = Path(datasets_module.__file__).parent / f"{config.data.dataset_name}.py"


datamodule = Seq2SeqDataModule(
    dataset_path=str(dataset_path),
    data_collator= DataCollatorWithPrompt(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
    ),
    batch_size=config.trainer.batch_size,
    num_workers=config.trainer.num_workers,
    training_transform_func=compose_left(*training_transforms),
    eval_transform_func=compose_left(*eval_transforms),
)
datamodule.prepare_data()

model.config.use_cache = False

trainer = SFTTrainer(
    model=model,
    train_dataset=datamodule._dataset["train"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()