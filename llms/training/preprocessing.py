from typing import Callable

from datasets import DatasetDict
from datasets.formatting.formatting import LazyBatch
from toolz import groupby
from transformers import BatchEncoding, PreTrainedTokenizer
from transformers.utils import PaddingStrategy


def preprocess_data(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    max_context_length: int = 256,
    max_target_length: int = 32,
    batch_size: int = 128,
    num_proc: int = 8,
    transform: Callable | None = None,
) -> DatasetDict:
    def preprocess_function(samples: LazyBatch) -> BatchEncoding:
        if transform:
            samples = transform(samples)

        fields_batch = [process_fields(fields) for fields in samples["fields"]]
        inputs = [
            f"context:\n{text}\n\nfields:\n{fields['field_names']}\n"
            for text, fields in zip(samples["text"], fields_batch)
        ]

        model_inputs = tokenizer(
            inputs,
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

    def process_fields(fields: dict[str, list[str]]) -> dict[str, str]:
        grouped = groupby(
            lambda x: x[0],
            seq=zip(fields["field_name"], fields["field_value"]),
        )
        unique_field_names = ", ".join(grouped)
        target = "\n".join(
            f"{field_name}: {value}"
            for field_name, values in grouped.items()
            for _, value in values
        )
        return {
            "field_names": unique_field_names,
            "target": target,
        }

    remove_columns = [
        col
        for col in dataset["train"].column_names
        if col not in ("input_ids", "attention_mask", "labels")
    ]
    tokenized_dataset = dataset.map(
        preprocess_function,
        remove_columns=remove_columns,
        num_proc=num_proc,
        batched=True,
        batch_size=batch_size,
    )

    return tokenized_dataset
