import json
from collections.abc import Iterable
from pathlib import Path
from random import randint

import datasets
from sklearn.model_selection import train_test_split

from llms import STORAGE_DIR
from llms.utils.dvc import maybe_get_dvc


class ZeroShotDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="zero-shot",
            version=datasets.Version("0.1.0"),
            description="ZeroShot dataset",
        ),
    ]
    DATASET_DVC_PATH = STORAGE_DIR / "datasets" / "zero-shot"

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "fields": datasets.features.Sequence(
                        {
                            "field_name": datasets.Value("string"),
                            "field_value": datasets.Value("string"),
                        }
                    ),
                }
            ),
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> list[datasets.SplitGenerator]:
        dataset_dir = maybe_get_dvc(
            self.DATASET_DVC_PATH,
        )

        file_paths = [
            path.stem for path in dataset_dir.joinpath("annotations").iterdir()
        ]
        train_paths, test_paths = train_test_split(
            file_paths, test_size=0.2, random_state=42
        )

        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={"filepaths": paths},
            )
            for split, paths in (
                (datasets.Split.TRAIN, train_paths),
                (datasets.Split.VALIDATION, test_paths),
            )
        ]

    def _generate_examples(
        self, filepaths: Iterable[Path]
    ) -> Iterable[tuple[str, dict]]:
        for name in filepaths:
            ground_truth_path = self.DATASET_DVC_PATH / "annotations" / f"{name}.json"
            fields = json.loads(ground_truth_path.read_text())["fields"]
            if not fields:
                continue

            ocr_path = self.DATASET_DVC_PATH / "ocr" / f"{name}.json"
            text = json.loads(ocr_path.read_text())["text"]

            item = {
                "id": name,
                "text": text,
                "fields": fields,
            }
            yield name, item
