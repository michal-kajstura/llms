from typing import Literal

from pydantic import BaseSettings


class PreprocessingConfig(BaseSettings):
    answer_delimiter: str = "|"
    line_delimiter: str = " | "
    tab_delimiter: str = " ; "
    max_context_length: int = 1024
    max_target_length: int = 512
    min_num_fields: int = 4
    max_num_fields: int = 16


class ModelConfig(BaseSettings):
    model_name: str = "google/flan-t5-large"
    load_in_kbit: Literal[4, 8, None] = None


class OptimizerConfig(BaseSettings):
    name: str = "adam"
    lr: float = 0.001


class DataModuleConfig(BaseSettings):
    dataset_name: str = "zero_shot"
    batch_size: int = 2
    num_workers: int = 0


class TrainerConfig(BaseSettings):
    max_epochs: int = 16
    accelerator: str = "cuda"
    accumulate_grad_batches: int = 8
    precision: str = "bf16"
    limit_val_batches: int = 64
    check_val_every_n_epoch: int = 2


class GenerationConfig(BaseSettings):
    temperature: float = 0.0


class MonitorConfig(BaseSettings):
    name: str = "extraction_match"
    mode: str = "max"


class MetricsConfig(BaseSettings):
    monitor: MonitorConfig = MonitorConfig()
    used_metrics: list[str] = ["anls", "extraction_match"]
    extraction_match: dict = {
        "similarity_threshold": 0.8,
        "case_sensitive": False,
        "line_delimiter": " | ",
    }


class PeftConfig(BaseSettings):
    config_type: str = "lora"
    lora: dict = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none"}
    prefix_tuning: dict = {"num_virtual_tokens": 20}


class TrainingConfig(BaseSettings):
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    datamodule: DataModuleConfig = DataModuleConfig()
    trainer: TrainerConfig = TrainerConfig()
    generation: GenerationConfig = GenerationConfig()
    metrics: MetricsConfig = MetricsConfig()
    peft: PeftConfig = PeftConfig()
    seed: int = 42
