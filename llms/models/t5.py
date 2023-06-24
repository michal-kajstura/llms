from __future__ import annotations

from typing import Any

from transformers import AutoModelForSeq2SeqLM

from llms.configs.training import T5ModelConfig
from llms.models.base import BaseLLMWrapper


class T5LLMWrapper(BaseLLMWrapper[T5ModelConfig]):
    @classmethod
    def _from_config(cls, config: T5ModelConfig, **kwargs: Any) -> T5LLMWrapper:
        return AutoModelForSeq2SeqLM.from_pretrained(
            config.model_name,
            device_map="auto",
            offload_folder="offload",
        )
