from transformers import pipeline, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
import random
import torch
from datasets import load_dataset
from statistics import mean

# ppo_dir = "./llama_ppo_step5000step_2400"
ppo_dir = "./checkpoints/checkpoint-1000"
sft_model_dir = "./LMFlow/output_models/finetuned_llama2"
base_dir = "../../llama/llama-2-7b"
device = "cuda" if torch.cuda.is_available() else "cpu"
# base_model = LlamaForCausalLM.from_pretrained(base_dir).to(device)
# base_tokenizer = LlamaTokenizer.from_pretrained(base_dir)
# base_tokenizer.pad_token_id=base_tokenizer.eos_token_id
# base_tokenizer.padding_side = 'left'
# sft_model = LlamaForCausalLM.from_pretrained(sft_model_dir).to(device)
# sft_tokenizer = LlamaTokenizer.from_pretrained(sft_model_dir)
# sft_tokenizer.pad_token_id=sft_tokenizer.eos_token_id
# sft_tokenizer.padding_side = 'left'
model = LlamaForCausalLM.from_pretrained(base_dir).to(device)
tokenizer = LlamaTokenizer.from_pretrained(base_dir)
tokenizer.pad_token = tokenizer.eos_token

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
test_dataset = load_dataset("Anthropic/hh-rlhf")["test"]["chosen"]
selected_items = random.sample(test_dataset, 1000)
batch_size = 5
num_batches = len(selected_items) // batch_size
print(num_batches)

def process_batch(batch):
    prompts = [text.split("Assistant:")[0].split("Human:")[1].strip() for text in batch]
    print(prompts)
    input_ids = tokenizer(prompts, return_tensors='pt', padding=True).input_ids.to(device)
    attention_mask = input_ids.attention_mask
    outputs = model.generate(input_ids,  attention_mask=attention_mask).to(device)
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    formatted_responses = ["###human: " + prompt + " ###assistant: " + generated_text[len(prompt):] for prompt, generated_text in zip(prompts, generated_texts)]
    pipe_outputs = rm_pipe(formatted_responses, **pipe_kwargs)
    rewards = [output[0]["score"] for output in pipe_outputs]
    print("batch_avg: {}".format(mean(rewards)))
    return rewards, formatted_responses

# Process all batches and calculate the average reward
all_rewards = []
all_responses = []
for i in range(num_batches):
    batch = selected_items[i*batch_size:(i+1)*batch_size]
    batch_rewards, samples = process_batch(batch)
    all_rewards.extend(batch_rewards)
    all_responses.extend(samples)
print(len(all_rewards))
average_reward = mean(all_rewards)
print("Average Reward:", average_reward)
print(all_responses[100])