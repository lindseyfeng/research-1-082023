# -*- coding: utf-8 -*_
#llama
from transformers import LLaMATokenizer,LlamaForCausalLM
#train
from transformers import Trainer, TrainingArguments
#dataset
from datasets import load_dataset
import torch

torch.cuda.empty_cache()
tokenizer = LLaMATokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Load the IMDB dataset
data = load_dataset("imdb")

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

tokenized_datasets = data.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

print(train_dataset)
print(test_dataset)

# Define training arguments and initialize Trainer
training_args = TrainingArguments(
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
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
trainer.save_model("YOUR_SAVED_MODEL_PATH_HERE")
