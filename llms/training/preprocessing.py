from datasets import DatasetDict
from transformers import PreTrainedTokenizer
from transformers.utils import PaddingStrategy


def preprocess_data(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    max_context_length: int = 256,
    max_target_length: int = 32,
    batch_size: int = 128,
    num_proc: int = 8,
) -> DatasetDict:
    def preprocess_function(samples):
        inputs = [
            f"context: {text}\n\nquestion: {question}\n\nanswer:"
            for text, question in zip(samples["text"], samples["question"])
        ]

        model_inputs = tokenizer(
            inputs,
            max_length=max_context_length,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=True,
        )

        labels = tokenizer(
            text_target=samples["answer"],
            max_length=max_target_length,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    remove_columns = [
        col
        for col in dataset["train"].column_names
        if col not in ("input_ids", "attention_mask", "labels")
    ]
    tokenized_dataset = dataset.map(
        preprocess_function,
        remove_columns=remove_columns,
        num_proc=num_proc,
        batched=True,
        batch_size=batch_size,
    )

    return tokenized_dataset
