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
saved_directory = "./t5_imdb_batch_reward/reward_checkpoint-1"
model = T5ForConditionalGeneration.from_pretrained(saved_directory)
tokenizer = T5TokenizerFast.from_pretrained(saved_directory)

prefix = "complete the following: "

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
      
def truncate_add_instruction_and_tokenize(batch):
    # Add prefix and truncate the first 64 tokens
    modified_texts = [prefix + ' '.join(text.split()[:64]) for text in batch['text']]
    input = tokenizer(modified_texts, truncation=True, padding='max_length', max_length=200, return_tensors="pt")
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
        _, text = heapq.heappop(self.queue)
        return text

    def peek(self):
        return self.queue[0][1]

    def __len__(self):
        return len(self.queue)

#bert model
# SentimentModel = SentimentModel(bert_model, 256, 1, 2, True, 0.25)
# bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# SentimentModel.load_state_dict(torch.load('model.pt', map_location=device))
# scaled_model = ModelWithTemperature(SentimentModel)
# scaled_model.load_state_dict(torch.load('model_with_temperature.pth', map_location=device))
# best_temperature = scaled_model.temperature.item()

if __name__ == "__main__":
    #infer from t5
    tokenized_datasets = dataset.map(truncate_add_instruction_and_tokenize, batched=True)
    print(tokenized_datasets)
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=800, collate_fn=collate_fn) 
    for batch in train_dataloader:
        count +=1
        with torch.no_grad(): 
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            pairs = []
            all_predictions = []
            all_scores = []
            pq = PriorityQueue()
            # Generate predictions
            if(count > 1):
                print(f"count: {count}")
                checkpoint_folder = f"./t5_imdb_batch/reward_checkpoint-{count-1}"
                model = T5ForConditionalGeneration.from_pretrained(checkpoint_folder)
                tokenizer = T5TokenizerFast.from_pretrained(checkpoint_folder)
            print(tokenizer)
            outputs = model.generate(input_ids, attention_mask=attention_mask, max_length = 48, min_length=48, eos_token_id=None)
            for inp_id, out in zip(input_ids, outputs):
                pairs.append((inp_id, out))
            for inp_id, out in pairs:
                input_text = tokenizer.decode(inp_id, skip_special_tokens=True)
                output_text = tokenizer.decode(out, skip_special_tokens=True)
                predicted_text = input_text + output_text
                all_predictions.append(predicted_text)
                reward_score = rm(predicted_text)
                # scaled_sentiment = predict_scaled_sentiment(scaled_model, bert_tokenizer, predicted_text, best_temperature)
                all_scores.append(reward_score)
                print(reward_score)
            for text, score in zip(all_predictions, all_scores):
            #     diverse_score = distinct_n_sentence_level(text,4)
            #     reward_score = rm(text)
            #     # print("text: {}, score: {}".format(text, reward_score))
            #     pq.push(text,0.5*diverse_score+0.5*score)
                
                # print("text: {}, score: {}".format(text, reward_score))
                pq.push(text,score)
        #train
        training_dataset = [pq.pop() for _ in range(160)] #100*0.2
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
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=64,
            num_train_epochs=1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir='./logs',
            logging_steps=128,
            do_train=True,
            do_eval=True,
            output_dir='./t5_imdb'
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
        checkpoint_folder = f"./t5_imdb_batch_reward/reward_checkpoint-{count}"
        trainer.save_model(checkpoint_folder)
        tokenizer.save_pretrained(checkpoint_folder)

    #save finetuned   
    trainer.save_model("./t5_imdb_complete")
    tokenizer.save_pretrained('./t5_imdb_complete')





        