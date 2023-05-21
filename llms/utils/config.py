from pathlib import Path
from typing import Any

from yaml import safe_load


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as file:
        return safe_load(file)
