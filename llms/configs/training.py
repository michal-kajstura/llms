import abc
from typing import Literal

from peft import TaskType
from pydantic import BaseSettings, root_validator


class DataConfig(BaseSettings):
    dataset_name: str = "zero_shot"
    answer_delimiter: str = "\n"
    line_delimiter: str = "\n"
    tab_delimiter: str = "\t"
    max_context_length: int = 1024
    max_target_length: int = 512
    min_num_fields: int = 4
    max_num_fields: int = 16
    prompt_template: str = "{text} Extract these fields: {field_names} "
    normalize_text: bool = True


class OptimizerConfig(BaseSettings):
    lr: float = 0.0005


class AdamWConfig(OptimizerConfig):
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


class SchedulerConfig(BaseSettings):
    num_training_steps: int = 10000


class LinearSchedulerWithWarmupConfig(SchedulerConfig):
    num_warmup_steps: int = 16


class ModelConfig(BaseSettings, abc.ABC):
    model_name: str
    load_in_kbit: Literal[4, 8, None] = None
    temperature: float = 0.0


class MPTModelConfig(ModelConfig):
    model_name: str = "mosaicml/mpt-7b-instruct"
    attn_impl: str = "triton"
    init_device: str = "cuda:0"


class T5ModelConfig(ModelConfig):
    model_name: str = "google/flan-t5-large"


class TrainerConfig(BaseSettings):
    max_epochs: int = 16
    accelerator: str = "cuda"
    accumulate_grad_batches: int = 8
    precision: str = "bf16-mixed"
    limit_val_batches: int = 64
    check_val_every_n_epoch: int = 2

    batch_size: int = 2
    num_workers: int = 8


class MetricsConfig(BaseSettings):
    main_metric: str = "extraction_match"
    mode: str = "max"
    used_metrics: list[str] = ["anls", "extraction_match"]
    extraction_match: dict = {
        "similarity_threshold": 0.8,
        "case_sensitive": False,
        "line_delimiter": "\n",
    }


class PeftConfigSettings(BaseSettings):
    task_type: TaskType = TaskType.SEQ_2_SEQ_LM


class LoraConfigSettings(PeftConfigSettings):
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    # "target_modules": ["q", "k", "v"],
    target_modules: str = ".*(SelfAttention|EncDecAttention).*(q|v|k)$"


class PrefixTuningConfigSettings(PeftConfigSettings):
    num_virtual_tokens: int = 20


class TrainingConfig(BaseSettings):
    seed: int = 42
    data: DataConfig = DataConfig()
    model: ModelConfig = T5ModelConfig()
    optimizer: OptimizerConfig = AdamWConfig()
    scheduler: SchedulerConfig = LinearSchedulerWithWarmupConfig()
    trainer: TrainerConfig = TrainerConfig()
    metrics: MetricsConfig = MetricsConfig()
    peft: PeftConfigSettings = LoraConfigSettings()

    @root_validator
    def _enforce_model_specific_formatting(cls, values: dict) -> dict:
        if 'T5' in values['model'].model_name:
            data = values['data']
            data.answer_delimiter = ' | '
            data.line_delimiter = ' | '
            data.tab_delimiter = ' '

        return values

    @root_validator
    def _set_line_delimiter(cls, values: dict) -> dict:
        values["metrics"].extraction_match["line_delimiter"] = values["data"].line_delimiter
        return values

