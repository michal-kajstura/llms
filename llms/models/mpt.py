from __future__ import annotations

from typing import Any

from transformers import AutoConfig, DataCollatorForLanguageModeling, PreTrainedTokenizer

from llms.configs.training import MPTModelConfig
from llms.models.base import BaseLLMWrapper
from llms.models.mpt_code.configuration_mpt import MPTConfig
from llms.models.mpt_code.modeling_mpt import MPTForCausalLM
from llms.training.collator import DataCollatorWithPrompt


class MPTLLMWrapper(BaseLLMWrapper[MPTModelConfig]):
    @classmethod
    def _from_config(cls, config: MPTModelConfig, **kwargs: Any) -> MPTLLMWrapper:
        model_config = MPTConfig.from_pretrained(config.model_name)
        model_config.attn_config['attn_impl'] = config.attn_impl
        model_config.init_device = config.init_device

        return MPTForCausalLM.from_pretrained(
            config.model_name,
            config=model_config,
            device_map="auto",
            offload_folder='offload',
        )

    @classmethod
    def _init_tokenizer(cls, config: MPTLLMWrapper) -> PreTrainedTokenizer:
        tokenizer = super()._init_tokenizer(config)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    @property
    def data_collator(self):
        return DataCollatorWithPrompt(
            tokenizer=self._tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

    def generate(self, *args, **kwargs):
        return self._model.generate(*args, **kwargs)
