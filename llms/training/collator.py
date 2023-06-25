import torch
from transformers import DataCollatorForLanguageModeling


class DataCollatorWithPrompt(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, pad_to_multiple_of=None, prefix_tokens=None, return_tensors=None):
        super().__init__(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
        )

    def __call__(
        self, features, return_tensors=None,
    ) -> dict[str, torch.Tensor]:
        features = [
            {
                "input_ids": example["input_ids"] + example["labels"],
                "attention_mask": example["attention_mask"],
            }
            for example in features
        ]

        return super().__call__(features, return_tensors=return_tensors)
