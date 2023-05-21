from typing import Any

import datasets
import evaluate
import numpy as np
import scipy
from Levenshtein import ratio


class ExtractionMatch(evaluate.Metric):
    def __init__(
        self,
        similarity_threshold: float = 0.8,
        case_sensitive: bool = False,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self._similarity_threshold = similarity_threshold
        self._case_sensitive = case_sensitive

    def _info(self):
        return evaluate.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            ),
        )

    def _compute(  # type: ignore[override]
        self,
        *,
        predictions: list[str],
        references: list[str],
    ) -> dict[str, Any]:

        total_score = 0.0
        total_references = 0
        for prediction, reference in zip(predictions, references):
            score, num_references = self._compute_metric(prediction, reference)
            total_score += score
            total_references += num_references
        score = total_score / total_references
        return {"extraction_match": score}

    def _compute_metric(self, prediction: str, reference: str) -> tuple[float, int]:
        if not self._case_sensitive:
            prediction = prediction.lower()
            reference = reference.lower()
        prediction_extractions = self._parse_text_as_extractions(prediction)
        reference_extractions = self._parse_text_as_extractions(reference)

        matrix = np.zeros((len(reference_extractions), len(prediction_extractions)))
        for prediction_idx, (prediction_field_name, prediction_field_value) in enumerate(
            prediction_extractions
        ):
            for reference_idx, (reference_field_name, reference_field_value) in enumerate(
                reference_extractions
            ):
                if prediction_field_name == reference_field_name:
                    matrix[reference_idx, prediction_idx] = ratio(
                        prediction_field_value, reference_field_value
                    )

        row_ids, col_ids = scipy.optimize.linear_sum_assignment(-matrix)
        total_score = 0.0
        for row_id, col_id in zip(row_ids, col_ids):
            total_score += matrix[row_id, col_id] > self._similarity_threshold

        return total_score, len(reference_extractions)

    @staticmethod
    def _parse_text_as_extractions(text: str) -> list[tuple[str, str]]:
        extractions = []
        for line in text.split('|'):
            parts = line.split(":", maxsplit=1)
            match parts:
                case [field_name, field_value]:
                    extractions.append((field_name, field_value.strip()))
                case _:
                    pass
        return extractions
