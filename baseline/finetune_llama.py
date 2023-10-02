# -*- coding: utf-8 -*_
#llama
from transformers import LLaMATokenizer,LlamaForCausalLM
#train
from transformers import Trainer, TrainingArguments
#dataset
from datasets import load_dataset
import torch

tokenizer = LLaMATokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")

# Load the IMDB dataset
data = load_dataset("imdb")

# Tokenize the data
train_encodings = tokenizer(data['train']['text'], truncation=True, padding=True)
val_encodings = tokenizer(data['test']['text'], truncation=True, padding=True)  # IMDB uses 'test' instead of 'validation'

# Prepare data for PyTorch
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_encodings, data['train']['label'])
val_dataset = CustomDataset(val_encodings, data['test']['label'])

# Define training arguments and initialize Trainer
training_args = TrainingArguments(
    per_device_train_batch_size=25000,
    per_device_eval_batch_size=25000,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=1000,
    do_train=True,
    do_eval=True,
    output_dir='./llama_7b_imdb'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Start training
trainer.train()

# Save the model
trainer.save_model("YOUR_SAVED_MODEL_PATH_HERE")
