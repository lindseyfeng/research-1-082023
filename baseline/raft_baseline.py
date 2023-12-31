# -*- coding: utf-8 -*_
from lib2to3.pgen2 import token
import sys
sys.path.append('../')  # Append the parent directory to sys.path
#bert
from transformers import BertModel, AutoTokenizer, DataCollatorForSeq2Seq
from bert.main import ModelWithTemperature, predict_scaled_sentiment, SentimentModel

#t5 finetune
from transformers import Trainer, TrainingArguments

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
#t5 inference
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from distinct_n.metrics import distinct_n_sentence_level
#reward
from reward_model_bert import BERTRewardModel

from statistics import mean 

#random seed
random.seed(42)

# count batch num
count = 0

# Get the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#load rm
rm = BERTRewardModel(lr = 2e-5)
rm.load_state_dict(torch.load('reward_model_0_complete.pt', map_location=device))

#load dataset
dataset = load_dataset("imdb")

#load t5 model
saved_directory = "./t5_imdb"
model = T5ForConditionalGeneration.from_pretrained(saved_directory)
tokenizer = T5TokenizerFast.from_pretrained(saved_directory)

prefix = "complete the following with a positive sentiment: "

#bert_model
bert_model = BertModel.from_pretrained('bert-base-uncased')

class LengthSampler:
    """
    Samples a length
    """

    def __init__(self, min_value, max_value):
        self.values = list(range(min_value, max_value))

    def __call__(self):
        return np.random.choice(self.values)
      
def truncate_add_instruction_and_tokenize(text):
    modified_text = prefix + ' '.join(text.split()[:64])
    input = tokenizer(modified_text, truncation=True, padding='max_length', max_length=200, return_tensors="pt")
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
    input_ids = torch.stack([item["input_ids"].clone().detach() for item in batch])
    attention_mask = torch.stack([item["attention_mask"].clone().detach() for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

#prepare data for finetune t5
def prepare_dataset(examples):
    length = LengthSampler(50, 60)
    split_ids = [length() for _ in range(len(examples["text"]))]
    token_ids = tokenizer(examples["text"], truncation=True, max_length=120 ,padding='max_length',)
    input_ids = [ids[:idx]+[tokenizer.eos_token_id] for idx, ids in zip(split_ids, token_ids["input_ids"])]
    label_ids = [ids[idx:] for idx, ids in zip(split_ids, token_ids["input_ids"])]
    print(label_ids)
    return {"input_ids": input_ids, "labels": label_ids}

#PQ for sample selection
class PriorityQueue:
    def __init__(self):
        self.queue = []

    def push(self, text, score):
        heapq.heappush(self.queue, (-score, text))  # Using negative score to simulate max-heap

    def pop(self):
        score, text = heapq.heappop(self.queue)
        return -score, text

    def peek(self):
        return self.queue[0][1]

    def __len__(self):
        return len(self.queue)

#bert model
SentimentModel = SentimentModel(bert_model, 256, 1, 2, True, 0.25)
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
SentimentModel.load_state_dict(torch.load('model.pt', map_location=device))
scaled_model = ModelWithTemperature(SentimentModel)
scaled_model.load_state_dict(torch.load('model_with_temperature.pth', map_location=device))
best_temperature = scaled_model.temperature.item()

if __name__ == "__main__":
    #infer from t5
    all_score = []
    negative_samples = dataset["train"].filter(lambda example: example['label'] == 0)["text"]
    sampled_negative_samples = random.sample(negative_samples, 1000)
    print(sampled_negative_samples[0])
    positive_samples = dataset["train"].filter(lambda example: example['label'] == 1)["text"]
    sampled_positive_samples = random.sample(positive_samples, 4000)
    print(sampled_positive_samples[0])
    combined_samples = sampled_negative_samples + sampled_positive_samples
    random.shuffle(combined_samples)
    tokenized_datasets = [truncate_add_instruction_and_tokenize(item) for item in combined_samples]
    print(tokenized_datasets[0])
    print(len(tokenized_datasets))
    train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=256, collate_fn=collate_fn) 
    for batch in train_dataloader:
        count +=1
        with torch.no_grad(): 
            input_ids = batch["input_ids"]
            print(len(input_ids))
            attention_mask = batch["attention_mask"]
            pairs = []
            training_dataset = []
            all_scores = []
            # Generate predictions

            if(count > 1):
                print(f"count: {count}")
                checkpoint_folder = f"./t5_imdb_batch/diverse_scorecheckpoint-{count-1}"
                model = T5ForConditionalGeneration.from_pretrained(checkpoint_folder)
                tokenizer = T5TokenizerFast.from_pretrained(checkpoint_folder)
            print(tokenizer)

            inner_count = 0
            for inp_id, mask in zip(input_ids, attention_mask):
                print(inner_count)
                if(inner_count % 100 == 0):
                    print("inner_count")
                inner_count += 1
                pq = PriorityQueue()
                input_text = tokenizer.decode(inp_id.view(-1).tolist(), skip_special_tokens=True)
                output = model.generate(inp_id, attention_mask=mask, max_length=48, min_length=48, eos_token_id=None, temperature=1.8, no_repeat_ngram_size=2, num_return_sequences=5, do_sample=True)
                for out in output:
                    output_text = tokenizer.decode(out, skip_special_tokens=True)
                    predicted_text = input_text + " " + output_text
                    scaled_sentiment = predict_scaled_sentiment(scaled_model, bert_tokenizer, predicted_text, best_temperature)
                    print(predicted_text)
                    print(scaled_sentiment)
                    # diverse_score = distinct_n_sentence_level(predicted_text, 4)
                    pq.push(predicted_text, scaled_sentiment)
                score, text = pq.pop()
                all_score.append(score)
                training_dataset.append(text)
                print(" ")
        #train
        print(training_dataset)
        dataset_dict = Dataset.from_dict({"text": training_dataset})
        tokenized_datasets_t5 = dataset_dict.map(prepare_dataset, batched=True)
        tokenized_datasets_t5 = tokenized_datasets_t5.remove_columns(["text"])
        tokenized_datasets_t5 = tokenized_datasets_t5.train_test_split(test_size=0.1)
        train_dataset = tokenized_datasets_t5["train"]
        test_dataset = tokenized_datasets_t5["test"]

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100)
        # Define training arguments and initialize Trainer
        training_args = TrainingArguments(
            per_device_train_batch_size=64,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=256,
            num_train_epochs=1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir='./logs',
            logging_steps=128,
            do_train=True,
            do_eval=True,
            learning_rate=1e-4,
            output_dir = "./t5_imdb_batch"
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
        checkpoint_folder = f"./t5_imdb_batch/diverse_scorecheckpoint-{count}"
        trainer.save_model(checkpoint_folder)
        tokenizer.save_pretrained(checkpoint_folder)
        print(mean(all_score))
    #save finetuned
    print("256, 1e-4")   
    print("./t5_imdb_completediverse_score")
    trainer.save_model("./t5_imdb_completediverse_score")
    tokenizer.save_pretrained('./t5_imdb_completediverse_score')





