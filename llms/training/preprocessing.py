from datasets.formatting.formatting import LazyBatch
from toolz import groupby
from transformers import BatchEncoding, PreTrainedTokenizer
from transformers.utils import PaddingStrategy


def preprocess_batch(
    samples: LazyBatch,
    tokenizer: PreTrainedTokenizer,
    max_context_length: int = 256,
    max_target_length: int = 32,
    answer_delimiter: str = "|",
    line_delimiter: str = "|",
    tab_delimiter: str = " ",
) -> BatchEncoding:
    fields_batch = [
        process_fields(
            fields,
            answer_delimiter=answer_delimiter,
        )
        for fields in samples["fields"]
    ]
    inputs = [
        prepare_prompt(
            text=text,
            fields=fields,
            line_delimiter=line_delimiter,
            tab_delimiter=tab_delimiter,
        )
        for text, fields in zip(samples["text"], fields_batch)
    ]

    model_inputs = tokenizer(
        text=inputs,
        max_length=max_context_length,
        padding=PaddingStrategy.DO_NOT_PAD,
        truncation=True,
    )
    labels = tokenizer(
        text_target=[fields["target"] for fields in fields_batch],
        max_length=max_target_length,
        padding=PaddingStrategy.DO_NOT_PAD,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def prepare_prompt(
    text: str,
    fields: dict[str, list[str]],
    line_delimiter: str = "|",
    tab_delimiter: str = " ",
) -> str:
    text = process_text(text, line_delimiter, tab_delimiter)
    field_names = ", ".join(fields["field_names"])
    return f"{text} Extract these fields: {field_names} "


def process_text(
    text: str,
    line_delimiter: str,
    tab_delimiter: str,
) -> str:
    text = text.replace("\n", line_delimiter)
    return text.replace("\t", tab_delimiter)


def process_fields(
    fields: dict[str, list[str]],
    answer_delimiter: str,
) -> dict[str, str | list[str]]:
    grouped = groupby(
        lambda x: x[0],
        seq=zip(fields["field_name"], fields["field_value"]),
    )
    unique_field_names = list(grouped.keys())
    target = answer_delimiter.join(
        f"{field_name}: {value}"
        for field_name, values in grouped.items()
        for _, value in values
    )
    return {
        "field_names": unique_field_names,
        "target": target,
    }
