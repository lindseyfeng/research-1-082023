from transformers import pipeline, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
import random
import torch
from datasets import load_dataset
from statistics import mean
import re
def split_first_qa(text):
    """
    Splits the conversation and extracts the first question from 'Human' and the first answer from 'Assistant'.
    """
    # Extract the first 'Human' part
    first_human_match = re.search(r'Human: (.*?)\s*(?=Assistant:|$)', text, re.DOTALL)
    first_human_part = first_human_match.group(1).strip() if first_human_match else None

    # Extract the first 'Assistant' part
    first_assistant_match = re.search(r'Assistant: (.*?)\s*(?=Human:|$)', text, re.DOTALL)
    first_assistant_part = first_assistant_match.group(1).strip() if first_assistant_match else None

    return "###human: " + first_human_part + " ###assistant: "  + first_assistant_part


rm_tokenizer = AutoTokenizer.from_pretrained("weqweasdas/hh_rlhf_rm_open_llama_3b")
device = "cuda" if torch.cuda.is_available() else "cpu"

rm_pipe = pipeline(
      "sentiment-analysis",
      model="weqweasdas/hh_rlhf_rm_open_llama_3b",
      device=device,
      tokenizer=rm_tokenizer,
      model_kwargs={"torch_dtype": torch.bfloat16}
  )

pipe_kwargs = {
      "return_all_scores": True,
      "function_to_apply": "none",
      "batch_size": 1
    }

random.seed(1111)
test_dataset = load_dataset("Anthropic/hh-rlhf")["test"]["rejected"]
selected_items = random.sample(test_dataset, 1000)
batch_size = 5
num_batches = len(selected_items) // batch_size
print(num_batches)

def process_batch(batch):
    prompts = [text.replace("Human:", "###Human:").replace("Assistant:", "###Assistant:")
 for text in batch]
    pipe_outputs = rm_pipe(prompts, **pipe_kwargs)
    rewards = [output[0]["score"] for output in pipe_outputs]
    print("batch_avg: {}".format(mean(rewards)))
    return rewards

# Process all batches and calculate the average reward
all_rewards = []
all_responses = []
for i in range(num_batches):
    batch = selected_items[i*batch_size:(i+1)*batch_size]
    batch_rewards = process_batch(batch)
    all_rewards.extend(batch_rewards)
print(len(all_rewards))
average_reward = mean(all_rewards)
print("Average Reward:", average_reward)
