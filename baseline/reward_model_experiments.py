import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformers import BertModel, BertTokenizer
from transformers import BertModel, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset, load_metric, Dataset
from torch.utils.data import DataLoader
sys.path.append('../') 
from transformers import BertModel, AutoTokenizer, DataCollatorForSeq2Seq
from reward_model_bert import BERTRewardModel
from distinct_n.metrics import distinct_n_sentence_level
from bert.main import ModelWithTemperature, predict_scaled_sentiment, SentimentModel

model = BERTRewardModel(lr = 0.001)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.load_state_dict(torch.load('reward_model_0.pt', map_location=device))
bert_model = BertModel.from_pretrained('bert-base-uncased')
SentimentModel = SentimentModel(bert_model, 256, 1, 2, True, 0.25)
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
SentimentModel.load_state_dict(torch.load('model.pt', map_location=device))
scaled_model = ModelWithTemperature(SentimentModel)
scaled_model.load_state_dict(torch.load('model_with_temperature.pth', map_location=device))
best_temperature = scaled_model.temperature.item()


if __name__ == "__main__":
    dataset = load_dataset("imdb")
    reward = []
    text_dataloader = DataLoader(dataset["train"].shuffle(seed=1111).select(range(256))['text'], batch_size=64, shuffle=True)
    for sentences in text_dataloader:
        for text in sentences:
            scaled_sentiment = predict_scaled_sentiment(scaled_model, bert_tokenizer, text, best_temperature)
            diverse_score = distinct_n_sentence_level(text,4)
            reward.append([scaled_sentiment, diverse_score])
        reward = torch.tensor(reward, dtype=torch.float32)
        
        sigmas = torch.Tensor(reward.std(0)) * sigma_mult
        total_loss = 0
        total_acc = 0
        total = 0
        reward_scale = []
        
        oss, acc, outs = model.get_loss(sentences, reward, sigmas)
        print("train_rm: loss: {}, acc: {}". format(loss, acc))



