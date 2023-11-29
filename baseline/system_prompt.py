# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

random.seed(42)

from datasets import load_dataset

dataset = load_dataset("Anthropic/hh-rlhf")["test"]["chosen"]
selected_items = random.sample(dataset, 3000)
for text in selected_items:

    dialogues = text.split("Assistant:")

    # Filter and clean up the parts spoken by the Human
    human_dialogues = [part.split("Human:")[1].strip() for part in dialogues if "Human:" in part]

    print(human_dialogues)
    
tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")