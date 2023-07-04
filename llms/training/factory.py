from accelerate import init_empty_weights
from peft import (
    LoraConfig,
    PeftConfig,
    PrefixTuningConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from llms.configs.training import (
    LoraConfigSettings,
    MPTModelConfig,
    PeftConfigSettings, PrefixTuningConfigSettings,
    T5ModelConfig,
    TrainingConfig,
)
from llms.models.base import BaseLLMWrapper
from llms.models.mpt import MPTLLMWrapper
from llms.models.t5 import T5LLMWrapper


def get_model(
    config: TrainingConfig,
) -> BaseLLMWrapper:
    match config.model:
        case MPTModelConfig() as model_config:
            wrapper = MPTLLMWrapper.from_config(model_config)
        case T5ModelConfig() as model_config:
            wrapper = T5LLMWrapper.from_config(model_config)
        case _:
            raise NotImplementedError(f"Model {config.model} not implemented")

    if config.peft is not None:
        peft_config = get_peft_config(config.peft)
        wrapper.model = _convert_to_peft(
            model=wrapper.model,
            peft_config=peft_config,
            load_in_kbit=config.model.load_in_kbit is not None,
        )

    wrapper.training_config = config
    return wrapper


def _convert_to_peft(
    model: PreTrainedModel,
    peft_config: PeftConfig,
    load_in_kbit: bool,
) -> PreTrainedModel:
    if load_in_kbit:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def get_peft_config(peft_config: PeftConfigSettings) -> PeftConfig | None:
    match peft_config:
        case LoraConfigSettings() as peft_config:
            return LoraConfig(
                task_type=peft_config.task_type,
                r=peft_config.r,
                lora_alpha=peft_config.lora_alpha,
                lora_dropout=peft_config.lora_dropout,
                bias=peft_config.bias,
                target_modules=peft_config.target_modules,
            )
        case PrefixTuningConfigSettings() as peft_config:
            return PrefixTuningConfig(
                num_virtual_tokens=peft_config.num_virtual_tokens,
                task_type=peft_config.task_type,
            )
        case None:
            return None
        case _:
            raise ValueError(f"Unknown peft_config_type: {peft_config}")
