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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BERTRewardModel.load_state_dict(torch.load('reward_model_0.pt', map_location=device))

if __name__ == "__main__":
    dataset = load_dataset("imdb")
    text_dataloader = DataLoader(dataset["train"].shuffle(seed=1111).select(range(256))['text'], batch_size=64, shuffle=True)
    for sentences in train_dataloader:
        oss, acc, outs = model.get_loss(sentences)
        print("train_rm: loss: {}, acc: {}". format(loss, acc))



