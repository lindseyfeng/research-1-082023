# -*- coding: utf-8 -*_
import sys
sys.path.append('../')  # Append the parent directory to sys.path
from bert.main import ModelWithTemperature, predict_scaled_sentiment

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
    modified_texts = [prefix + ' '.join(tokenizer.tokenize(text)[64:]) for text in batch['text']]
    input = tokenizer(modified_texts, truncation=True, padding='max_length', max_length=512, return_tensors="pt")
    print(type(input['input_ids']))
    print(input['input_ids'].shape)
    return input


#prepoocess llama imdb
def preprocess_function(examples):
  return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_preds):
    metric = load_metric("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



if __name__ == "__main__":
  #infer from t5
  if INFER:
    saved_directory = "./t5_imdb"
    model = T5ForConditionalGeneration.from_pretrained(saved_directory)
    tokenizer = T5TokenizerFast.from_pretrained(saved_directory)
    tokenized_datasets = dataset.map(truncate_add_instruction_and_tokenize, batched=True)
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=1280)
    with torch.no_grad():  # Ensure no gradients are computed
      for batch in train_dataloader:
          input_ids = batch["input_ids"]
          attention_mask = batch["attention_mask"]
          # Generate predictions
          outputs = model.generate(input_ids, attention_mask=attention_mask, max_length = 48)
          # Decode predictions to get the text
          predicted_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
          # Now you can print or process the predicted_texts as required
          print(predicted_texts)
          # model.load_state_dict(torch.load('model.pt', map_location=device))
          # scaled_model = ModelWithTemperature(model)
          # scaled_model.load_state_dict(torch.load('model_with_temperature.pth', map_location=device))
          # best_temperature = scaled_model.temperature.item()
          # scaled_sentiment = predict_scaled_sentiment(scaled_model, tokenizer, predicted_texts, best_temperature)
          # print(scaled_sentiment)
