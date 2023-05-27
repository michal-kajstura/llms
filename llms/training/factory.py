from peft import (
    LoraConfig, PeftConfig, PrefixTuningConfig, TaskType, get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from llms.configs.training import TrainingConfig


def get_model(
    config: TrainingConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    load_in_kbit_args = {}
    match config.model.load_in_kbit:
        case 8:
            load_in_kbit_args["load_in_8bit"] = True
        case 4:
            load_in_kbit_args["load_in_4bit"] = True
        case _:
            pass

    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.model.model_name,
        trust_remote_code=True,
        **load_in_kbit_args,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

    peft_config = get_peft_config(config)
    if peft_config is not None:
        model = _convert_to_peft(
            model=model,
            peft_config=peft_config,
            load_in_kbit=bool(load_in_kbit_args),
        )

    return model, tokenizer


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
