from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from llms import DATASETS_PATH
from llms.training.factory import get_model
from llms.training.preprocessing import preprocess_data

dataset_path = DATASETS_PATH / "docvqa.py"
dataset = load_dataset(str(dataset_path))

model_name = "google/flan-t5-large"
model, tokenizer = get_model(
    model_name=model_name,
    load_in_8bit=True,
    peft_config=LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    ),
)

tokenized_dataset = preprocess_data(
    dataset=dataset,
    tokenizer=tokenizer,
    max_context_length=512,
    max_target_length=64,
)


save_path = Path(f"data/{model_name.replace('/', '-')}")

training_args = Seq2SeqTrainingArguments(
    output_dir=str(save_path),
    per_device_train_batch_size=2,
    learning_rate=1e-3,
    num_train_epochs=5,
    logging_dir=f"{save_path}/logs",
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="no",
    report_to="mlflow",
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    pad_to_multiple_of=8,
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
)
trainer.train()
