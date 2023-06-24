import json
from collections.abc import Iterable
from pathlib import Path

import datasets
from toolz import groupby

from llms import STORAGE_DIR
from llms.utils.dvc import maybe_get_dvc


class DocVQA(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="docvqa",
            version=datasets.Version("1.0.0"),
            description="DocVQA dataset",
        ),
    ]
    DATASET_DVC_PATH = STORAGE_DIR / "datasets" / "docvqa"

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "words_boxes": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "bounding_box": datasets.features.Sequence(
                                datasets.Value("float32"),
                            ),
                        }
                    ),
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        datasets.Value("string"),
                    ),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> list[datasets.SplitGenerator]:
        dataset_dir = maybe_get_dvc(
            self.DATASET_DVC_PATH,
        )

        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={"filepath": dataset_dir / str(split)},
            )
            for split in (
                datasets.Split.TRAIN,
                datasets.Split.VALIDATION,
            )
        ]

    def _generate_examples(self, filepath: Path) -> Iterable[tuple[str, dict]]:
        ground_truth_path = filepath / "ground_truth.json"
        with ground_truth_path.open("r") as file:
            ground_truth = json.load(file)

        documents = groupby(
            lambda example: example["image"],
            ground_truth["data"],
        )

        for image_name, examples in documents.items():
            document_name = Path(image_name).stem
            json_path = filepath / "ocr_results" / f"{document_name}.json"
            with json_path.open("r") as file:
                ocr_results = json.load(file)
            context, words_boxes = self._transform_ocr_results(ocr_results)

            for example in examples:
                id_ = example["questionId"]
                item = {
                    "id": id_,
                    "text": context,
                    "words_boxes": words_boxes,
                    "question": example["question"],
                    "answer": example["answers"][0],
                    "answers": example["answers"],
                }
                yield id_, item

    @staticmethod
    def _transform_ocr_results(ocr_results: dict) -> tuple[str, list[dict]]:
        def transform_bbox(bbox: list[int]) -> tuple[float, float, float, float]:
            left = min(bbox[0::2])
            top = min(bbox[1::2])
            right = max(bbox[0::2])
            bottom = max(bbox[1::2])
            return (
                left / image_width,
                top / image_height,
                right / image_width,
                bottom / image_height,
            )

        image_width = ocr_results["recognitionResults"][0]["width"]
        image_height = ocr_results["recognitionResults"][0]["height"]

        lines = [line for page in ocr_results["recognitionResults"] for line in page["lines"]]

        raw_lines = "\n".join(line["text"] for line in lines)

        words_boxes = [
            {
                "text": word["text"],
                "bounding_box": transform_bbox(word["boundingBox"]),
            }
            for line in lines
            for word in line["words"]
        ]

        return raw_lines, words_boxes
