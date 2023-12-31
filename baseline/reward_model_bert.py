import sys
sys.path.append('../') 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformers import BertModel, BertTokenizer
from bert.main import ModelWithTemperature, predict_scaled_sentiment, SentimentModel
from transformers import BertModel, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset, load_metric, Dataset
from torch.utils.data import DataLoader
from distinct_n.metrics import distinct_n_sentence_level

device = 'cuda' if torch.cuda.is_available() else 'cpu'
bert_model = BertModel.from_pretrained('bert-base-uncased')
SentimentModel = SentimentModel(bert_model, 256, 1, 2, True, 0.25)
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
SentimentModel.load_state_dict(torch.load('model.pt', map_location=device))
scaled_model = ModelWithTemperature(SentimentModel)
scaled_model.load_state_dict(torch.load('model_with_temperature.pth', map_location=device))
best_temperature = scaled_model.temperature.item()


def train_rm(rm, train_dataloader, bsz=16, sigma_mult=1):
    total_loss = 0
    total_acc = 0
    total = 0
    reward_scale = []
    for sentences in train_dataloader:
        reward = []
        for text in sentences:
            scaled_sentiment = predict_scaled_sentiment(scaled_model, bert_tokenizer, text, best_temperature)
            diverse_score = distinct_n_sentence_level(text,4)
            reward.append([scaled_sentiment, diverse_score])
        reward = torch.tensor(reward, dtype=torch.float32)
        
        sigmas = torch.Tensor(reward.std(0)) * sigma_mult

        idx = np.random.choice(len(sentences), bsz)
        batch_sentences = [sentences[i] for i in idx]
        loss, acc, outs = rm.get_loss(batch_sentences, reward[idx], sigmas)
        if loss <= 0:
            continue
        print("train_rm: loss: {}, acc: {}". format(loss, acc))
        print("count: {}". format(total))
        reward_scale += outs
        rm.optimizer.zero_grad()
        loss.backward()
        rm.optimizer.step()

        total_loss += loss.item()
        total_acc += acc
        total += 1
    print(total_loss / (total + 1e-5), total_acc / (total + 1e-5))

    rm.mu, rm.sigma = np.array(reward_scale).mean(), np.array(reward_scale).std()

    return total_loss / (total + 1e-5), total_acc / (total + 1e-5)

def test_rm(rm, test_dataloader):
    total_loss = 0
    total_acc = 0
    total = 0
    reward_scale = []
    for sentences in test_dataloader:
        print(len(sentences))
        reward = []
        for text in sentences:
            scaled_sentiment = predict_scaled_sentiment(scaled_model, bert_tokenizer, text, best_temperature)
            diverse_score = distinct_n_sentence_level(text,4)
            reward.append([scaled_sentiment, diverse_score])
        reward = torch.tensor(reward, dtype=torch.float32)
        sigmas = torch.Tensor(reward.std(0))
        idx = np.random.choice(len(sentences), 16)
        batch_sentences = [sentences[i] for i in idx]
        loss, acc, outs = rm.get_loss(batch_sentences, reward[idx], sigmas) 
        if loss <= 0:
            continue
        total_loss += loss.item()
        total_acc += acc
        total += 1
    print(total_loss / (total + 1e-5), total_acc / (total + 1e-5))
    return total_loss / (total + 1e-5), total_acc / (total + 1e-5)




class BERTRewardModel(nn.Module):
    def __init__(self, lr, normalize=False, bert_model_name='bert-base-uncased'):
        super(BERTRewardModel, self).__init__()

        # Load the pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        # Use BERT's hidden size to define the FC layer
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

        self.mu = 0
        self.sigma = 1
        self.normalize = normalize

        self.heirarchy = [0, 1] # Now we only have two metrics: sentiment and diversity

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, sentences):
        # Tokenize sentences and get BERT's output
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.bert(**inputs)
        
        # Use the [CLS] token representation to predict the reward
        cls_representation = outputs.last_hidden_state[:, 0, :]
        reward = self.fc(cls_representation)

        if self.normalize:
            reward = (reward - self.mu) / self.sigma
        return reward


    def get_loss(self, x, reward_signal, sigmas):
        sign = [1, 1] # One sign for each metric: sentiment and diversity
        total_loss = 0
        total = 0
        correct = 0
        outs = []
        print(len(x))

        for i in range(len(x)):
            for j in np.random.choice(len(x), 5, replace=False):
                x_i = x[i]
                x_j = x[j]

                reward_i = self(x_i)
                if j == 0:
                    outs.append(reward_i.item())
                reward_j = self(x_j)
                # print(reward_i)
                # print(reward_j)

                reward_info_i = reward_signal[i]
                reward_info_j = reward_signal[j]
                # print(reward_info_i)
                # print(reward_info_j)
                # reward_i.requires_grad_(True)
                # reward_j.requires_grad_(True)
                # print(reward_i.requires_grad, reward_j.requires_grad)


                # Level 1: sentiment_Score
                if reward_info_i[self.heirarchy[0]] * sign[0] > reward_info_j[self.heirarchy[0]] * sign[0] + sigmas[self.heirarchy[0]]:
                    loss = -1 * torch.log(torch.sigmoid(reward_i - reward_j))
                    # print("loss1: {}, {}".format(loss, reward_i - reward_j))
                    if reward_i > reward_j:
                        correct += 1
                elif reward_info_j[self.heirarchy[0]] * sign[0] > reward_info_i[self.heirarchy[0]] * sign[0] + sigmas[self.heirarchy[0]]:
                    loss = -1 * torch.log(torch.sigmoid(reward_j - reward_i))
                    # print("loss2: {}, {}".format(loss, reward_i - reward_j))
                    if reward_j > reward_i:
                        correct += 1
                # Level 2: diversity_Score
                elif reward_info_i[self.heirarchy[1]] * sign[1] > reward_info_j[self.heirarchy[1]] * sign[1] + sigmas[self.heirarchy[1]]:
                    loss = -1 * torch.log(torch.sigmoid(reward_i - reward_j))
                    # print("loss3: {}, {}".format(loss, reward_i - reward_j))
                    if reward_i > reward_j:
                        correct += 1
                elif reward_info_j[self.heirarchy[1]] * sign[1] > reward_info_i[self.heirarchy[1]] * sign[1] + sigmas[self.heirarchy[1]]:
                    loss = -1 * torch.log(torch.sigmoid(reward_j - reward_i))
                    # print("loss4: {}, {}".format(loss, reward_i - reward_j))
                    if reward_j > reward_i:
                        correct += 1
                else:
                    continue
                total += 1
                total_loss += loss
                # total_loss.requires_grad_(True)

        return total_loss / (total + 1e-5), correct / (total + 1e-5), outs


if __name__ == "__main__":
    #train a reward mdoel
    dataset = load_dataset("imdb")
    RewardModel = BERTRewardModel(lr = 2e-5)
    for i in range(3):
        text_dataloader = DataLoader(dataset["train"].shuffle(seed=42).select(range(2500))['text'], batch_size=64, shuffle=True)
        test_dataloader = DataLoader(dataset["test"].shuffle(seed=1111).select(range(192))['text'], batch_size=64, shuffle=True)
        loss, acc = train_rm(RewardModel, text_dataloader)
        print("loss: {}, acc: {}".format(loss, acc))

        test_loss, test_acc = test_rm(RewardModel, test_dataloader)
        print("test_loss: {}, test_acc: {}".format(test_loss, test_acc))
        if(test_acc >= 0.8):
            torch.save(RewardModel.state_dict(), f'reward_model_{i}_complete.pt')
            break
        torch.save(RewardModel.state_dict(), f'reward_model_{i}.pt')
    torch.save(RewardModel.state_dict(), 'reward_model.pt')
       
