from __future__ import annotations

import abc
from typing import Literal, Union

from peft import TaskType
from pydantic import BaseSettings


class DataConfig(BaseSettings):
    dataset_name: str = "zero_shot"
    answer_delimiter: str = "\n"
    line_delimiter: str = "\n"
    tab_delimiter: str = "\t"
    max_context_length: int = 1024
    max_target_length: int = 512
    min_num_fields: int = 4
    max_num_fields: int = 16
    prompt_template: str = "{text}\n{field_names}"
    normalize_text: bool = True


class OptimizerConfig(BaseSettings):
    lr: float = 0.0001


class AdamWConfig(OptimizerConfig):
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


class SchedulerConfig(BaseSettings):
    pass


class LinearSchedulerWithWarmupConfig(SchedulerConfig):
    num_warmup_steps: int = 16


class ModelConfig(BaseSettings, abc.ABC):
    model_name: str
    load_in_kbit: Literal[4, 8, None] = None
    temperature: float = 0.0

    def modify_config(self, config: TrainingConfig) -> TrainingConfig:
        return config


class MPTModelConfig(ModelConfig):
    model_name: str = "mosaicml/mpt-7b-instruct"
    # model_name: str = "tiiuae/falcon-7b"
    attn_impl: str = "triton"
    init_device: str = "cuda:0"

    def modify_config(self, config: TrainingConfig):
        config.data.answer_delimiter = "\n"
        config.data.line_delimiter = "\n"
        config.data.tab_delimiter = "  "  # 2 spaces
        config.metrics.extraction_match["line_delimiter"] = "\n"

        if isinstance(peft_config := config.peft, LoraConfigSettings):
            peft_config.target_modules = ["Wqkv"]

        config.data.prompt_template = (
            "{text}\n"
            "### Instruction: Extract following fields {field_names} from the text.\n"
            "### Answer:\n"
        )

        config.peft.task_type = TaskType.CAUSAL_LM
        return config


class FalconModelConfig(ModelConfig):
    model_name: str = "tiiuae/falcon-7b"

    def modify_config(self, config: TrainingConfig):
        config.data.answer_delimiter = "\n"
        config.data.line_delimiter = "\n"
        config.data.tab_delimiter = "  "  # 2 spaces
        config.metrics.extraction_match["line_delimiter"] = "\n"

        if isinstance(peft_config := config.peft, LoraConfigSettings):
            peft_config.target_modules = ["key_query_value"]

        config.data.prompt_template = (
            "{text}\n"
            ">>QUESTION<< Extract following fields {field_names} from the text.\n"
            ">>ANSWER<<\n"
        )

        config.peft.task_type = TaskType.CAUSAL_LM
        return config


class T5ModelConfig(ModelConfig):
    model_name: str = "google/flan-t5-large"

    def modify_config(self, config: TrainingConfig):
        config.data.answer_delimiter = " | "
        config.data.line_delimiter = " | "
        config.data.tab_delimiter = "  "
        config.metrics.extraction_match["line_delimiter"] = " | "

        if isinstance(peft_config := config.peft, LoraConfigSettings):
            peft_config.target_modules = ".*(SelfAttention|EncDecAttention).*(q|v|k)$"

        config.data.prompt_template = (
            "{text}\n"
            "### Instruction: Extract following fields {field_names} from the text. "
        )

        config.peft.task_type = TaskType.SEQ_2_SEQ_LM

        return config


class TrainerConfig(BaseSettings):
    max_epochs: int = 16
    accelerator: str = "cuda"
    precision: Literal["16-mixed", "bf16-mixed", "32"] = "bf16-mixed"
    check_val_every_n_epoch: int = 1

    batch_size: int = 2
    accumulate_grad_batches: int = 8
    num_workers: int = 12


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
    target_modules: Union[str, list[str]] = ["q", "k", "v"]


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
    peft: Union[PeftConfigSettings, None] = LoraConfigSettings()
