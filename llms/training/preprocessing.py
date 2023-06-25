from collections import Counter
from random import randint, shuffle
from typing import Any

from cytoolz import groupby
from datasets.formatting.formatting import LazyBatch
from text_unidecode import unidecode
from transformers import BatchEncoding, PreTrainedTokenizer
from transformers.utils import PaddingStrategy


def create_prompt(
    text: str,
    line_delimiter: str,
    tab_delimiter: str,
    prompt_template: str,
    fields: list[dict[str, str]] | None = None,
    field_names: list[str] | None = None,
    normalize_text: bool = False,
) -> str:
    if fields and field_names:
        raise ValueError("Only one of fields or field_names can be specified")

    text = _preprocess_text(
        text=text,
        line_delimiter=line_delimiter,
        tab_delimiter=tab_delimiter,
        normalize_text=normalize_text,
    )

    if fields:
        field_names = [field["name"] for field in fields]

    field_names_str = ", ".join(field_names)

    return prompt_template.format(
        text=text,
        field_names=field_names_str,
    )


def _preprocess_text(
    text: str | None,
    line_delimiter: str = "\n",
    tab_delimiter: str = "\t",
    normalize_text: bool = False,
) -> str | None:
    if text is None:
        return None

    text = text.replace("\n", line_delimiter)
    text = text.replace("\t", tab_delimiter)

    if normalize_text:
        text = unidecode(text)
    return text


class PreprocessBatch:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompt_template: str,
        max_context_length: int | None = None,
        max_target_length: int | None = None,
        answer_delimiter: str = "\n",
        line_delimiter: str = "\n",
        tab_delimiter: str = "\t",
        normalize_text: bool = False,
    ) -> None:
        self._tokenizer = tokenizer
        self._prompt_template = prompt_template
        self._max_context_length = max_context_length
        self._max_target_length = max_target_length
        self._answer_delimiter = answer_delimiter
        self._line_delimiter = line_delimiter
        self._tab_delimiter = tab_delimiter
        self._normalize_text = normalize_text

    def __call__(self, samples: LazyBatch) -> BatchEncoding:
        fields_batch = samples["fields"]
        text_batch = samples["text"]

        if self._normalize_text:
            fields_batch = [
                {
                    'name': [_preprocess_text(field, normalize_text=True) for field in fields['name']],
                    'value': [_preprocess_text(field, normalize_text=True) for field in fields['value']]
                }
                for fields in fields_batch
            ]
            text_batch = [
                _preprocess_text(
                    text=text,
                    normalize_text=True,
                    line_delimiter=self._line_delimiter,
                    tab_delimiter=self._tab_delimiter,
                )
                for text in text_batch
            ]

        fields_batch = [
            self._process_fields(
                fields,
            )
            for fields in fields_batch
        ]
        inputs = [
            create_prompt(
                text=text,
                field_names=fields["field_names"],
                prompt_template=self._prompt_template,
                line_delimiter=self._line_delimiter,
                tab_delimiter=self._tab_delimiter,
            )
            for text, fields in zip(text_batch, fields_batch)
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
        print('labels in transform ', len(model_inputs["labels"][0]))
        return model_inputs

    def _process_fields(
        self,
        fields: dict[str, list[str]],
    ) -> dict[str, str | list[str]]:
        grouped = groupby(
            lambda x: x[0],
            seq=zip(fields["name"], fields["value"]),
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
            counter = Counter(fields["name"])
            ids = [i for i, v in enumerate(fields["name"]) if counter[v] == 1]
            fields = {k: [v[i] for i in ids] for k, v in fields.items()}

        all_fields = len(fields["name"])
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
