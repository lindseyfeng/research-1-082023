# -*- coding: utf-8 -*_
#train
from transformers import Trainer, TrainingArguments
#dataset
from datasets import load_dataset
import torch

# Load model directly
from transformers import LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# Load model directly

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")

print(model)
print(tokenizer)

# Load the  dataset
dataset = load_dataset("Anthropic/hh-rlhf")
MAX_TOKENS = 256

def preprocess_function(examples):
    print(examples)
    inputs = examples["chosen"]
    # Tokenize the inputs
    model_inputs = tokenizer(inputs, max_length=MAX_TOKENS, truncation=True)
    # Filter out examples that exceed the max token limit
    filtered_input_ids = []
    filtered_labels = []
    for input_ids, target in zip(model_inputs["input_ids"], targets):
        if len(input_ids) <= MAX_TOKENS:
            filtered_input_ids.append(input_ids)
            # Tokenize and add the corresponding target
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(target, max_length=MAX_TOKENS, truncation=True)
            filtered_labels.append(labels["input_ids"])

    return {"input_ids": filtered_input_ids, "labels": filtered_labels}

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
print(tokenized_dataset)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./rlhf_finetuned_model",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# Train the model
trainer.train()
trainer.save_model("./vicuna_imdb")
tokenizer.save_pretrained('./vicuna_imdb')