from collections import Counter

from cytoolz import groupby
from datasets.formatting.formatting import LazyBatch
from random import randint, shuffle
from transformers import BatchEncoding, PreTrainedTokenizer
from transformers.utils import PaddingStrategy
from typing import Any


class PreprocessBatch:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_context_length: int | None = None,
        max_target_length: int | None = None,
        answer_delimiter: str = "|",
        line_delimiter: str = "|",
        tab_delimiter: str = " ",
    ) -> None:
        self._tokenizer = tokenizer
        self._max_context_length = max_context_length
        self._max_target_length = max_target_length
        self._answer_delimiter = answer_delimiter
        self._line_delimiter = line_delimiter
        self._tab_delimiter = tab_delimiter

    def __call__(self, samples: LazyBatch) -> BatchEncoding:
        fields_batch = [
            self._process_fields(
                fields,
            )
            for fields in samples["fields"]
        ]
        inputs = [
            self._prepare_prompt(
                text=text,
                fields=fields,
            )
            for text, fields in zip(samples["text"], fields_batch)
        ]

        model_inputs = self._tokenizer(
            text=inputs,
            max_length=self._max_context_length,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=True,
        )
        labels = self._tokenizer(
            text_target=[fields["target"] for fields in fields_batch],
            max_length=self._max_target_length,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def _prepare_prompt(
        self,
        text: str,
        fields: dict[str, list[str]],
    ) -> str:
        text = self._process_text(text)
        field_names = ", ".join(fields["field_names"])
        return f"{text} Extract these fields: {field_names} "

    def _process_text(
        self,
        text: str,
    ) -> str:
        text = text.replace("\n", self._line_delimiter)
        return text.replace("\t", self._tab_delimiter)

    def _process_fields(
        self,
        fields: dict[str, list[str]],
    ) -> dict[str, str | list[str]]:
        grouped = groupby(
            lambda x: x[0],
            seq=zip(fields["field_name"], fields["field_value"]),
        )
        unique_field_names = list(grouped.keys())
        target = self._answer_delimiter.join(
            f"{field_name}: {value}"
            for field_name, values in grouped.items()
            for _, value in values
        )
        return {
            "field_names": unique_field_names,
            "target": target,
        }


class TransformFields:
    def __init__(
        self,
        remove_multiple_occurrences: bool = True,
        min_num_fields: int | None = None,
        max_num_fields: int | None = None,
    ) -> None:
        self._remove_multiple_occurrences = remove_multiple_occurrences
        self._min_num_fields = min_num_fields
        self._max_num_fields = max_num_fields

    def __call__(self, samples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        keys = list(samples.keys())
        num_samples = len(samples[keys[0]])
        samples = [{k: samples[k][i] for k in keys} for i in range(num_samples)]
        transformed = [self._transform(sample) for sample in samples]
        return {k: [sample[k] for sample in transformed] for k in keys}

    def _transform(self, sample: dict[str, Any]) -> dict[str, Any]:
        fields = sample["fields"]

        if self._remove_multiple_occurrences:
            counter = Counter(fields["field_name"])
            ids = [i for i, v in enumerate(fields["field_name"]) if counter[v] == 1]
            fields = {k: [v[i] for i in ids] for k, v in fields.items()}

        all_fields = len(fields["field_name"])
        num_fields = randint(
            min(self._min_num_fields or 0, all_fields),
            min(self._max_num_fields or all_fields, all_fields),
        )
        ids = list(range(all_fields))
        shuffle(ids)
        ids = ids[:num_fields]

        fields = {k: [v[i] for i in ids] for k, v in fields.items()}

        return {
            **sample,
            "fields": fields,
        }
