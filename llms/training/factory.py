from typing import Any

from peft import (LoraConfig, PeftConfig, TaskType, get_peft_model,
                  prepare_model_for_int8_training)
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          PreTrainedModel, PreTrainedTokenizer)


def get_model(
    model_name: str,
    load_in_8bit: bool = True,
    peft_config: PeftConfig | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        load_in_8bit=load_in_8bit,
        device_map="auto",
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


def get_peft_config(config: dict[str, Any]) -> PeftConfig | None:
    peft_config_type = config["peft_config_type"]
    match peft_config_type:
        case "lora":
            peft_config = config["lora"]
            return LoraConfig(
                r=peft_config["r"],
                lora_alpha=peft_config["lora_alpha"],
                target_modules=["q", "v"],
                lora_dropout=peft_config["lora_dropout"],
                bias=peft_config["bias"],
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
        case None:
            return None
        case _:
            raise ValueError(f"Unknown peft_config_type: {peft_config_type}")
