from transformers import pipeline, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
import random
import torch
from datasets import load_dataset
from statistics import mean

sft_model_dir = "./LMFlow/output_models/finetuned_llama2"
base_dir = "../../llama/llama-2-7b"
ppo_dir = "/llama_ppo_step5000step_2000"

base_model = LlamaForCausalLM.from_pretrained(base_dir)
base_tokenizer = LlamaTokenizer.from_pretrained(base_dir)
# sft_model = LlamaForCausalLM.from_pretrained(sft_model_dir)
# sft_tookenizer = LlamaTokenizer.from_pretrained(sft_model_dir)
# ppo_model = LlamaForCausalLM.from_pretrained(ppo_dir)
# ppo_tokenizer = LlamaTokenizer.from_pretrained(ppo_dir)

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

random.seed(42)
test_dataset = load_dataset("Anthropic/hh-rlhf")["test"]["chosen"]
selected_items = random.sample(test_dataset, 1000)
batch_size = 20
num_batches = len(selected_items) // batch_size
print(num_batches)

def process_batch(batch):
    prompts = [text.split("Assistant:")[0].split("Human:")[1].strip() for text in batch]
    input_ids = base_tokenizer(prompts, padding=True, pad_token_id=base_tokenizer.eos_token_id, return_tensors='pt').input_ids.to(device)
    outputs = base_model.generate(input_ids, min_length = 200, max_length=600, pad_token_id=base_tokenizer.eos_token_id).to(device)
    generated_texts = base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
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
