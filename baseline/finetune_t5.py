# -*- coding: utf-8 -*_
#train
from transformers import Trainer, TrainingArguments
#dataset
from datasets import load_dataset
import torch

# Load model directly
from transformers import T5TokenizerFast, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
import numpy as np

class LengthSampler:
    """
    Samples a length
    """

    def __init__(self, min_value, max_value):
        self.values = list(range(min_value, max_value))

    def __call__(self):
        return np.random.choice(self.values)


tokenizer = T5TokenizerFast.from_pretrained("t5-large", model_max_length=512)
model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")

# Load the IMDB dataset
data = load_dataset("imdb")
data["train"] = data["train"].filter(lambda example: example['label'] == 1)


length = LengthSampler(4, 128)

def prepare_dataset(examples):
  split_ids = [length() for _ in range(len(examples["text"]))]
  token_ids = tokenizer(examples["text"], truncation=True, max_length=512)
  input_ids = [ids[:idx]+[tokenizer.eos_token_id] for idx, ids in zip(split_ids, token_ids["input_ids"])]
  label_ids = [ids[idx:] for idx, ids in zip(split_ids, token_ids["input_ids"])]

  return {"input_ids": input_ids, "labels": label_ids}


tokenized_datasets = data.map(prepare_dataset, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text", "label"])
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(5000))

print(train_dataset)
print(test_dataset)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100)
# Define training arguments and initialize Trainer
training_args = TrainingArguments(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=128,
    do_train=True,
    do_eval=True,
    output_dir='./t5_imdb'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator
)

# Start training
trainer.train()

# Save the model
trainer.save_model("./t5_imdb_baseline")
tokenizer.save_pretrained('./t5_imdb_baseline')
