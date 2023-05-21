from typing import Any

import datasets
import evaluate
from Levenshtein import ratio


class ANLS(evaluate.Metric):
    def __init__(
        self,
        distance_threshold: float = 0.5,
        case_sensitive: bool = False,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self._distance_threshold = distance_threshold
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
        references: list[str | list[str]],
    ) -> dict[str, Any]:
        anls_score = 0.0
        for prediction, ground_truths in zip(predictions, references, strict=True):
            max_value = 0
            ground_truths = (
                [ground_truths] if isinstance(ground_truths, str) else ground_truths
            )

            if not self._case_sensitive:
                prediction = prediction.lower()
                ground_truths = [ground_truth.lower() for ground_truth in ground_truths]

            score = max(
                ratio(prediction, ground_truth) for ground_truth in ground_truths
            )
            anls_score += score if score > self._distance_threshold else 0.0

        anls_score /= len(predictions)
        return {"anls_score": anls_score}
