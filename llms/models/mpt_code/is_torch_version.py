import sys
import logging
import operator as op
from packaging import version
from packaging.version import Version, parse
from typing import Union
import importlib.util

# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

STR_OPERATION_TO_FUNC = {">": op.gt, ">=": op.ge, "==": op.eq, "!=": op.ne, "<=": op.le, "<": op.lt}

logger = logging.getLogger(__name__)

_torch_available = importlib.util.find_spec("torch") is not None
if _torch_available:
  try:
    _torch_version = importlib_metadata.version("torch")
    logger.info(f"PyTorch version {_torch_version} available.")
  except importlib_metadata.PackageNotFoundError:
    _torch_available = False

# This function was copied from: https://github.com/huggingface/accelerate/blob/874c4967d94badd24f893064cc3bef45f57cadf7/src/accelerate/utils/versions.py#L319
def compare_versions(library_or_version: Union[str, Version], operation: str, requirement_version: str):
  """
  Args:
  Compares a library version to some requirement using a given operation.
    library_or_version (`str` or `packaging.version.Version`):
      A library name or a version to check.
    operation (`str`):
      A string representation of an operator, such as `">"` or `"<="`.
    requirement_version (`str`):
      The version to compare the library version against
  """
  if operation not in STR_OPERATION_TO_FUNC.keys():
    raise ValueError(f"`operation` must be one of {list(STR_OPERATION_TO_FUNC.keys())}, received {operation}")
  operation = STR_OPERATION_TO_FUNC[operation]
  if isinstance(library_or_version, str):
    library_or_version = parse(importlib_metadata.version(library_or_version))
  return operation(library_or_version, parse(requirement_version))

# This function was copied from: https://github.com/huggingface/accelerate/blob/874c4967d94badd24f893064cc3bef45f57cadf7/src/accelerate/utils/versions.py#L338
def is_torch_version(operation: str, version: str):
  """
  Args:
  Compares the current PyTorch version to a given reference with an operation.
    operation (`str`):
      A string representation of an operator, such as `">"` or `"<="`
    version (`str`):
      A string version of PyTorch
  """
  return compare_versions(parse(_torch_version), operation, version)