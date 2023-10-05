# -*- coding: utf-8 -*_
import sys
sys.path.append('../')  # Append the parent directory to sys.path
#bert
from bert.main import ModelWithTemperature, predict_scaled_sentiment, SentimentModel
from bert.config import *
#t5 finetune
from finetune_t5 import LengthSampler 
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
from datasets import load_dataset, load_metric, Dataset
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
saved_directory = "./t5_imdb"
model = T5ForConditionalGeneration.from_pretrained(saved_directory)
tokenizer = T5TokenizerFast.from_pretrained(saved_directory)

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

#DataLoader collate
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

#prepare data for finetune t5
def prepare_dataset(examples):
    length = LengthSampler(4, 128)
    split_ids = [length() for _ in range(len(examples["text"]))]
    token_ids = tokenizer(examples["text"], truncation=True, max_length=512)
    input_ids = [ids[:idx]+[tokenizer.eos_token_id] for idx, ids in zip(split_ids, token_ids["input_ids"])]
    label_ids = [ids[idx:] for idx, ids in zip(split_ids, token_ids["input_ids"])]
    return {"input_ids": input_ids, "labels": label_ids}

#PQ for sample selection
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

#bert model
SentimentModel = SentimentModel(
  bert_model,
  HIDDEN_DIM,
  OUTPUT_DIM,
  N_LAYERS,
  BIDIRECTIONAL,
  DROPOUT
)

if __name__ == "__main__":
  #infer from t5
  all_predictions = []
  all_scores = []
  tokenized_datasets = dataset.map(truncate_add_instruction_and_tokenize, batched=True)
  print(tokenized_datasets["train"])
  train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=100, collate_fn=collate_fn)
  sample_batch = next(iter(train_dataloader))
  for batch in train_dataloader:
    with torch.no_grad(): 
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
          # Generate predictions
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length = 48)
        for output in outputs:
            predicted_text = tokenizer.decode(output, skip_special_tokens=True)
            all_predictions.append(predicted_text)
            SentimentModel.load_state_dict(torch.load('model.pt', map_location=device))
            scaled_model = ModelWithTemperature(SentimentModel)
            scaled_model.load_state_dict(torch.load('model_with_temperature.pth', map_location=device))
            best_temperature = scaled_model.temperature.item()
            scaled_sentiment = predict_scaled_sentiment(scaled_model, tokenizer, predicted_text, best_temperature)
            print(scaled_sentiment)
            all_scores.append(-scaled_sentiment)
            pq = PriorityQueue()
            for text, score in zip(all_predictions, all_scores):
                pq.push(text, score)
    #train
    training_dataset = [pq.pop() for _ in range(20)]
    dataset = Dataset.from_dict({"text": training_dataset})
    tokenized_datasets = dataset.map(prepare_dataset, batched=True)



        

