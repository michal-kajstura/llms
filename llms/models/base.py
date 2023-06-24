from __future__ import annotations

import abc
from typing import Any, Generic, TypeVar

from transformers import PreTrainedModel

from llms.configs.training import ModelConfig

TModel = TypeVar("TModel", bound=PreTrainedModel)
TConfig = TypeVar("TConfig", bound=ModelConfig)


class BaseLLMWrapper(abc.ABC, Generic[TConfig]):
    def __init__(self, model: PreTrainedModel) -> None:
        self._model = model

    @property
    def model(self) -> TModel:
        return self._model

    @classmethod
    def from_config(cls, config: TConfig) -> BaseLLMWrapper:
        load_in_kbit_args = {}
        match config.load_in_kbit:
            case 8:
                load_in_kbit_args["load_in_8bit"] = True
            case 4:
                load_in_kbit_args["load_in_4bit"] = True
            case _:
                pass

        model = cls._from_config(config, **load_in_kbit_args)
        return cls(model)

    @classmethod
    @abc.abstractmethod
    def _from_config(cls, config: TConfig, **kwargs: Any) -> PreTrainedModel:
        pass

    def generate(self, *args, **kwargs):
        return self._model.generate(*args, **kwargs)
