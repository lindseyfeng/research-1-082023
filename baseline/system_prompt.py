from transformers import pipeline
import torch
import random
from datasets import load_dataset
from transformers import AutoModel

torch.backends.cuda.matmul.allow_tf32 = True

pipe = pipeline("text-generation", model="lmsys/vicuna-7b-v1.5")

random.seed(42)

dataset = load_dataset("Anthropic/hh-rlhf")["test"]["chosen"]
selected_items = random.sample(dataset, 3000)

# Path to your finetuned model
model_path = 'output_models/finetuned_vicuna'

# Load the model
model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

reward = []
for text in selected_items:

    dialogues = text.split("Assistant:")

    # Filter and clean up the parts spoken by the Human
    human_dialogues = [part.split("Human:")[1].strip() for part in dialogues if "Human:" in part]
    string = ""

    for dialogue in human_dialogues:
        result = pipe(dialogue)[generated_text]
        string.append(result)

    print(string)
    tokenized_string = tokenizer(string)
    score = model(tokenized_string)
    print(score)
    reward.append(score)
