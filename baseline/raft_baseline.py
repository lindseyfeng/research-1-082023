# -*- coding: utf-8 -*_

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

from transformers import T5TokenizerFast, T5ForConditionalGeneration
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
  #infer from t5
  if INFER:
    saved_directory = "./t5_imdb"
    model = T5ForConditionalGeneration.from_pretrained(saved_directory)
    tokenizer = T5TokenizerFast.from_pretrained(saved_directory)
    input_text = "summarize: The quick brown fox jumps over the lazy dog."
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    # Generate output
    output_ids = model.generate(input_ids)
    # Decode the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(generated_text)