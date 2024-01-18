from transformers import pipeline, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
import random
import torch
from datasets import load_dataset
from statistics import mean
from trl import PPOTrainer, AutoModelForCausalLMWithValueHead

ppo_dir = "./llama_ppo_step5000step_2400"
# ppo_dir = "./checkpoints/checkpoint-1000"
# ppo_dir = "./finetuned_llama_ppostep_1800"
# ppo_dir = "./LMFlow/output_models/finetuned_llama2"
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
model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_dir).to(device)
tokenizer = LlamaTokenizer.from_pretrained(ppo_dir)
tokenizer.pad_token_id=tokenizer.eos_token_id
# tokenizer.padding_side = 'left'

rm_tokenizer = AutoTokenizer.from_pretrained("weqweasdas/hh_rlhf_rm")
device = "cuda" if torch.cuda.is_available() else "cpu"

rm_pipe = pipeline(
      "sentiment-analysis",
      model="weqweasdas/hh_rlhf_rm",
      device=device,
      tokenizer=rm_tokenizer,
      model_kwargs={"torch_dtype": torch.bfloat16}
  )

pipe_kwargs = {
      "return_all_scores": True,
      "function_to_apply": "none",
      "batch_size": 1
    }

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": -1,
    "max_new_tokens": 100,
}
ppo_trainer = PPOTrainer(model)
random.seed(1111)
test_dataset = load_dataset("Anthropic/hh-rlhf")["test"]["chosen"]
selected_items = random.sample(test_dataset, 1000)
batch_size = 5
num_batches = len(selected_items) // batch_size
print(num_batches)
def process_batch(batch):
    prompts = [text.split("Assistant:")[0].split("Human:")[1].strip() for text in batch]
    input_ids = tokenizer(prompts, padding=True, return_tensors='pt').input_ids.to(device)
    outputs = PPOTrainer.generate(input_ids, 
    return_prompt=False,
     **generation_kwargs,).to(device)
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    formatted_responses = ["###human: " + prompt + " ###assistant: " + generated_text[len(prompt):] for prompt, generated_text in zip(prompts, generated_texts)]
    pipe_outputs = rm_pipe(formatted_responses, **pipe_kwargs)
    rewards = [output[0]["score"] for output in pipe_outputs]
    print("batch_avg: {}".format(mean(rewards)))
    print(formatted_responses)
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
print(all_responses)
