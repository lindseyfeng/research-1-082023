# -*- coding: utf-8 -*_
# #llama
# from transformers import LLaMATokenizer,LlamaForCausalLM
#train
from transformers import Trainer, TrainingArguments
#dataset
from datasets import load_dataset
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-70b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Load the IMDB dataset
data = load_dataset("imdb")

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=510)

tokenized_datasets = data.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10000))
test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(5000))

print(train_dataset)
print(test_dataset)

# Define training arguments and initialize Trainer
training_args = TrainingArguments(
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=50,
    do_train=True,
    do_eval=True,
    output_dir='./llama_7b_imdb'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Start training
trainer.train()

# Save the model
trainer.save_model("llama_7b_imdb.pt")
