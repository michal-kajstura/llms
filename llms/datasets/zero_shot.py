import json
from collections.abc import Iterable
from pathlib import Path

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
                            "name": datasets.Value("string"),
                            "value": datasets.Value("string"),
                        }
                    ),
                }
            ),
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> list[datasets.SplitGenerator]:
        dataset_dir = maybe_get_dvc(self.DATASET_DVC_PATH)

        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={"filepaths": dataset_dir.joinpath(str(split)).iterdir()},
            )
            for split in (datasets.Split.TRAIN, datasets.Split.VALIDATION)
        ]

    def _generate_examples(
        self, filepaths: Iterable[Path]
    ) -> Iterable[tuple[str, dict]]:
        for path in filepaths:
            annotation = json.loads(path.read_text())
            fields = annotation["fields"]
            if not fields:
                continue

            name = path.stem
            text = annotation["text"]
            item = {
                "id": name,
                "text": text,
                "fields": fields,
            }
            yield name, item
