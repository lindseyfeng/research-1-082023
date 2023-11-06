import sys
sys.path.append('../') 
from transformers import BertModel, AutoTokenizer, DataCollatorForSeq2Seq
from reward_model_bert import BERTRewardModel

model = BERTRewardModel.load_state_dict(torch.load('reward_model_0.pt', map_location=device))

if __name__ == "__main__":
    dataset = load_dataset("imdb")
    text_dataloader = DataLoader(dataset["train"].shuffle(seed=1111).select(range(256))['text'], batch_size=64, shuffle=True)
    for sentences in train_dataloader:
        oss, acc, outs = model.get_loss(sentences)
        print("train_rm: loss: {}, acc: {}". format(loss, acc))



