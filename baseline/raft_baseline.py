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

def tokenize_for_infer(texts):
    # Tokenize the texts
    token_ids = tokenizer(texts, truncation=True, max_length=128, return_tensors="pt")
    input_ids = []
    for ids in token_ids["input_ids"]:
        # Find the position of the EOS token
        eos_position = (ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        # If EOS token is found, truncate everything after it. Otherwise, keep the whole sequence.
        input_ids.append(ids[:eos_position[0] + 1] if eos_position.numel() > 0 else ids)
    return input_ids




if __name__ == "__main__":
  #infer from t5
  if INFER:
    saved_directory = "./t5_imdb"
    model = T5ForConditionalGeneration.from_pretrained(saved_directory)
    tokenizer = T5TokenizerFast.from_pretrained(saved_directory)
    input_text = ["complete the following: A dog eats a pretty", "complete the following: A cat eats a pretty"]
    tokenized_inputs = tokenize_for_infer(input_text)
    print(tokenized_inputs)
    # Feed tokenized inputs to the model for generation
    output_ids = model.generate(tokenized_inputs[0], max_new_tokens=50)  # Adjust max_new_tokens as per your requirements

    # Decode the generated output
    generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
    print(generated_texts)