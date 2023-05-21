from collections import Counter
from random import randint, shuffle
from typing import Any


class TrainingTransform:
    def __init__(
        self,
        remove_multiple_occurrences: bool = True,
        min_num_fields: int = 1,
        max_num_fields: int = 16,
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
            min(self._min_num_fields, all_fields),
            min(self._max_num_fields, all_fields),
        )
        ids = list(range(all_fields))
        shuffle(ids)
        ids = ids[:num_fields]

        fields = {k: [v[i] for i in ids] for k, v in fields.items()}

        return {
            **sample,
            "fields": fields,
        }
