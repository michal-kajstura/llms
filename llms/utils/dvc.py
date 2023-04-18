import time
from logging import getLogger
from pathlib import Path

from llms import STORAGE_DIR

_LOGGER = getLogger(__name__)


class DVCLockError(IOError):
    pass


def get_dvc_path(
    target: str,
) -> Path:
    return Path(STORAGE_DIR, target)


def maybe_get_dvc(
    target: str,
    skip_exists: bool = True,
    wait_for_lock: bool = True,
    recursive: bool = False,
) -> Path:
    from dvc.lock import LockError
    from dvc.repo import Repo

    target_location = get_dvc_path(target=target)

    if skip_exists and target_location.exists():
        _LOGGER.info("Target %s exists - skipping", target)
        return target_location.absolute().resolve()

    repo = Repo(STORAGE_DIR)
    while True:
        try:
            stats = repo.pull(
                targets=[str(target_location)],
                recursive=recursive,
            )
            _LOGGER.info("Download %s : %s", target, stats)
        except LockError as ex:
            if not wait_for_lock:
                raise DVCLockError from ex
            _LOGGER.warning(ex)
            time.sleep(10)
        except Exception as error:
            raise ValueError(f"Failed downloading {target_location}") from error
        else:
            break

    if not target_location.exists():
        raise RuntimeError("File does not exist after completed download")

    return target_location.resolve()
