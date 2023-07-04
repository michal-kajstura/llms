from sklearn.model_selection import train_test_split

from pathlib import Path
from llms import DATASETS_PATH

# dataset_dir = DATASETS_PATH / "zero-shot"
dataset_dir = Path('/mnt/data/work/projects/tyro-datasets/data/zero-shot/evaluation/')

paths = list(dataset_dir.joinpath("annotations").iterdir())
val_paths, test_paths = train_test_split(
    paths,
    test_size=0.5,
    random_state=42,
)

val_path = dataset_dir / "val"
val_path.mkdir(exist_ok=True)
for path in val_paths:
    path.rename(val_path / path.name)

test_path = dataset_dir / "test"
test_path.mkdir(exist_ok=True)
for path in test_paths:
    path.rename(test_path / path.name)
