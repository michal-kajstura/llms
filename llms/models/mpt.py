from __future__ import annotations

from typing import Any

from transformers import AutoConfig, AutoModelForCausalLM

from llms.configs.training import MPTModelConfig
from llms.models.base import BaseLLMWrapper


class MPTLLMWrapper(BaseLLMWrapper[MPTModelConfig]):
    @classmethod
    def _from_config(cls, config: MPTModelConfig, **kwargs: Any) -> MPTLLMWrapper:
        config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
        config.attn_config['attn_impl'] = config.attn_impl
        config.init_device = config.init_device

        return AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            device_map="auto",
            offload_folder='offload',
        )

    def generate(self, *args, **kwargs):
        return self._model.generate(*args, **kwargs)
