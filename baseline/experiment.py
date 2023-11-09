import sys
import random
from typing import OrderedDict
sys.path.append('../')
#distinct 1,2
from distinct_n.metrics import distinct_n_corpus_level
#bert
from transformers import BertModel, AutoTokenizer
from bert.main import ModelWithTemperature, predict_scaled_sentiment, SentimentModel
#t5
from transformers import T5TokenizerFast, T5ForConditionalGeneration
#dataset
#huggingface dataset
import torch
from datasets import load_dataset, load_metric, Dataset
from torch.utils.data import DataLoader
from statistics import mean 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = load_dataset("imdb")

saved_directory = "./t5_imdb_batch/reward_checkpoint-11"
model = T5ForConditionalGeneration.from_pretrained(saved_directory)
tokenizer = T5TokenizerFast.from_pretrained(saved_directory)
print(tokenizer)
print(saved_directory)
bert_model = BertModel.from_pretrained('bert-base-uncased')
SentimentModel = SentimentModel(bert_model, 256, 1, 2, True, 0.25)
SentimentModel.load_state_dict(torch.load('model.pt', map_location=device))
scaled_model = ModelWithTemperature(SentimentModel)
scaled_model.load_state_dict(torch.load('model_with_temperature.pth', map_location=device))
best_temperature = scaled_model.temperature.item()
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

prefix = "complete the following: "

def truncate_add_instruction_and_tokenize(batch):
    # Add prefix and truncate the first 64 tokens
    modified_texts = [prefix + ' '.join(text.split()[:64]) for text in batch['text']]
    input = tokenizer(modified_texts, truncation=True, padding='max_length', max_length=64, return_tensors="pt")
    return input

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

if __name__ == "__main__":
    all_predictions = []
    all_scores = []
    pairs = []
    count = 0
    tokenized_datasets = dataset.map(truncate_add_instruction_and_tokenize, batched=True)
    test_samples = list(sample for sample in tokenized_datasets["train"]) #.filter(lambda example: example['label'] == 0))
    print(test_samples)
    random_test_samples = random.sample(test_samples, 100) #100 testing sample
    train_dataloader = DataLoader(random_test_samples, shuffle=True, batch_size=10, collate_fn=collate_fn) #100
    for batch in train_dataloader:
        with torch.no_grad(): 
            print(count)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            outputs = model.generate(input_ids, attention_mask=attention_mask, max_length = 48, min_length=48, eos_token_id=None)
            for inp_id, out in zip(input_ids, outputs):
                pairs.append((inp_id, out))
            for inp_id, out in pairs:
                input_text = tokenizer.decode(inp_id, skip_special_tokens=True)
                output_text = tokenizer.decode(out, skip_special_tokens=True)
                predicted_text = input_text + output_text
                all_predictions.append(predicted_text)
                print("input: {}".format(input_text))
                print("output: {}".format(output_text))
                scaled_sentiment = predict_scaled_sentiment(scaled_model, bert_tokenizer, output_text, best_temperature)
                all_scores.append(scaled_sentiment)
                print(scaled_sentiment)
            count += 1

    #DIVERSE 1,2
    print(all_predictions)
    SAMPLE_TIMES = 3200  # Number of samples
    line_list = all_predictions
    #DISTINCT 1, 2
    distinct_2 = distinct_n_corpus_level(all_predictions,2)
    distinct_3 = distinct_n_corpus_level(all_predictions,3)
    distinct_4 = distinct_n_corpus_level(all_predictions,4)
    distinct_5 = distinct_n_corpus_level(all_predictions,5)
    distinct_6 = distinct_n_corpus_level(all_predictions,6)
    distinct_7 = distinct_n_corpus_level(all_predictions,7)
    print("DISTINCT-2", distinct_2)
    print("DISTINCT-3", distinct_3)
    print("DISTINCT-4", distinct_4)
    print("DISTINCT-5", distinct_5)
    print("DISTINCT-6", distinct_6)
    print("DISTINCT-7", distinct_7)


    #average sentiment score
    average_sentiment = mean(all_scores)
    print("sentiment score", average_sentiment)




