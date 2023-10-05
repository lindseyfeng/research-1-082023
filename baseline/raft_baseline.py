# -*- coding: utf-8 -*_
import sys
sys.path.append('../')  # Append the parent directory to sys.path
from bert.main import ModelWithTemperature, predict_scaled_sentiment
#priority queue for sample selection
import heapq
# NN library
import torch
import torch.nn as nn
import torch.optim as optim

# Numerical computation
import numpy as np
# standard library
import random
import time
# Configuration
from config import *
#huggingface dataset
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import T5TokenizerFast, T5ForConditionalGeneration,AutoModelForSeq2SeqLM
# Set random seed for reproducible experiments

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Get the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#load dataset
dataset = load_dataset("imdb")

#load t5 model

prefix = "complete the following: "

def truncate_add_instruction_and_tokenize(batch):
    # Add prefix and truncate the first 64 tokens
    modified_texts = [prefix + ' '.join(text.split()[:64]) for text in batch['text']]
    input = tokenizer(modified_texts, truncation=True, padding='max_length', max_length=512, return_tensors="pt")
    return input


#prepoocess llama imdb
def preprocess_function(examples):
  return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_preds):
    metric = load_metric("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def collate_fn(batch):
    # Convert lists to tensors
    input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
    attention_mask = torch.stack([torch.tensor(item["attention_mask"]) for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": labels
    }

class PriorityQueue:
    def __init__(self):
        self.queue = []

    def push(self, text, score):
        heapq.heappush(self.queue, (-score, text))  # Using negative score to simulate max-heap

    def pop(self):
        _, text = heapq.heappop(self.queue)
        return text

    def peek(self):
        return self.queue[0][1]

    def __len__(self):
        return len(self.queue)


if __name__ == "__main__":
  #infer from t5
  if INFER:
    all_predictions = []
    saved_directory = "./t5_imdb"
    model = T5ForConditionalGeneration.from_pretrained(saved_directory)
    tokenizer = T5TokenizerFast.from_pretrained(saved_directory)
    tokenized_datasets = dataset.map(truncate_add_instruction_and_tokenize, batched=True)
    print(tokenized_datasets["train"])
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=100, collate_fn=collate_fn)
    sample_batch = next(iter(train_dataloader))
    with torch.no_grad():  # Ensure no gradients are computed
      for batch in train_dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
          # Generate predictions
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length = 48)
        for output in outputs:
          predicted_text = tokenizer.decode(output, skip_special_tokens=True)
          model.load_state_dict(torch.load('bert/model.pt', map_location=device))
          scaled_model = ModelWithTemperature(model)
          scaled_model.load_state_dict(torch.load('bert/model_with_temperature.pth', map_location=device))
          best_temperature = scaled_model.temperature.item()
          scaled_sentiment = predict_scaled_sentiment(scaled_model, tokenizer, predicted_text, best_temperature)
          print(scaled_sentiment)
