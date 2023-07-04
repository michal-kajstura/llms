from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
from transformers import (
    AutoTokenizer,
    DefaultDataCollator,
    GenerationConfig, PreTrainedModel,
    PreTrainedTokenizer,
)

from llms.configs.training import ModelConfig, TrainingConfig

TModel = TypeVar("TModel", bound=PreTrainedModel)
TConfig = TypeVar("TConfig", bound=ModelConfig)


class BaseLLMWrapper(abc.ABC, Generic[TConfig]):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.training_config: TrainingConfig | None = None

    @property
    def model(self) -> TModel:
        return self._model

    @model.setter
    def model(self, model: TModel) -> None:
        self._model = model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    @property
    @abc.abstractmethod
    def data_collator(self) -> DefaultDataCollator:
        pass

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
        tokenizer = cls._init_tokenizer(config)
        return cls(model, tokenizer)

    @classmethod
    @abc.abstractmethod
    def _from_config(cls, config: TConfig, **kwargs: Any) -> PreTrainedModel:
        pass

    @classmethod
    def _init_tokenizer(cls, config: TConfig) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(config.model_name)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: GenerationConfig,
    ) -> torch.Tensor:
        return self._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )

    def save_pretrained(self, path: str) -> None:
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)
        if self.training_config is not None:
            data_config = self.training_config.data
            config_path = Path(path) / 'training_config.json'
            with config_path.open('w') as f:
                json.dump(data_config.dict(), f)
