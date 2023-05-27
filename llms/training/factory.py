from peft import (LoraConfig, PeftConfig, PrefixTuningConfig, TaskType,
                  get_peft_model, prepare_model_for_int8_training)
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          PreTrainedModel, PreTrainedTokenizer)

from llms.configs.training import TrainingConfig


def get_model(
    model_name: str,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    peft_config: PeftConfig | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if peft_config is not None:
        model = _convert_to_peft(
            model=model,
            peft_config=peft_config,
            load_in_8bit=load_in_8bit,
        )

    return model, tokenizer


def _convert_to_peft(
    model: PreTrainedModel,
    peft_config: PeftConfig,
    load_in_8bit: bool = True,
) -> PreTrainedModel:
    if load_in_8bit:
        model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def get_peft_config(config: TrainingConfig) -> PeftConfig | None:
    peft_config_type = config.peft.config_type
    match peft_config_type:
        case "lora":
            peft_config = config.peft.lora
            return LoraConfig(
                r=peft_config["r"],
                lora_alpha=peft_config["lora_alpha"],
                target_modules=["q", "v"],
                lora_dropout=peft_config["lora_dropout"],
                bias=peft_config["bias"],
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
        case "prefix_tuning":
            peft_config = config.peft.prefix_tuning
            return PrefixTuningConfig(
                num_virtual_tokens=peft_config["num_virtual_tokens"],
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
        case None:
            return None
        case _:
            raise ValueError(f"Unknown peft_config_type: {peft_config_type}")
