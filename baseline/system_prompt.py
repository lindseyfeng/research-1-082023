# Load model directly
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="lmsys/vicuna-7b-v1.5")
import random

random.seed(42)

from datasets import load_dataset

pipe = pipeline("text-generation", model="lmsys/vicuna-7b-v1.5")

dataset = load_dataset("Anthropic/hh-rlhf")["test"]["chosen"]
selected_items = random.sample(dataset, 3000)

# reward_model = 
for text in selected_items:

    dialogues = text.split("Assistant:")

    # Filter and clean up the parts spoken by the Human
    human_dialogues = [part.split("Human:")[1].strip() for part in dialogues if "Human:" in part]

    for dialogue in human_dialogues:
        result = pipe(dialogue)
        print(result)

    print(human_dialogues)
