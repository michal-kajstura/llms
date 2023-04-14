from peft import (
    LoraConfig,
    PeftConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_int8_training,
)
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


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
