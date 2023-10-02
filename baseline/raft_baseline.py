# -*- coding: utf-8 -*_

# NN library
import torch
import torch.nn as nn
import torch.optim as optim

from transformers import DataCollatorWithPadding, evaluate, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Numerical computation
import numpy as np
# standard library
import random
import time
# Configuration
from config import *
#huggingface dataset
from datasets import load_dataset, load_metric
#llama7b
from transformers import pipeline,LLaMATokenizer,LlamaForCausalLM

# Set random seed for reproducible experiments
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Get the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#load dataset
dataset = load_dataset("imdb")

#load llama pipeline
generator = pipeline(model="decapoda-research/llama-7b-hf", device=device)

#load llama model
tokenizer = LLaMATokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")



# Tokensize and crop sentence to 510 (for 1st and last token) instead of 512 (i.e. `max_input_len`)
def tokenize_and_crop(sentence):
  tokens = tokenizer.tokenize(sentence, max_length=64, truncation=True)
  return tokens

#prepoocess llama imdb
def preprocess_function(examples):
  return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_preds):
    metric = load_metric("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



if __name__ == "__main__":
  #finetune llama
  if FINETUNE:
    # load data
    tokenized_imdb = imdb.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
      output_dir="llama_pretrained",
      learning_rate=2e-6,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=16,
       num_train_epochs=2,
       weight_decay=0.01,
       evaluation_strategy="epoch",
       save_strategy="epoch",
       load_best_model_at_end=True,
       push_to_hub=True,
      )
    
    trainer = Trainer(model=model, tokenizer=tokenizer,
                  data_collator=data_collator,
                  args=training_args,
                  train_dataset=tokenized_imdb["train"],
                  eval_dataset=tokenized_imdb["test"], 
                  compute_metrics=compute_metrics)
    trainer.train()
  
  # Infer from llama
  elif INFER:
    tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")