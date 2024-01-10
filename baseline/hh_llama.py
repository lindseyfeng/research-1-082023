import torch
from transformers import ( AutoModelForCausalLM, AutoTokenizer,
pipeline,
)
from peft import LoraConfig, PeftModel 
from trl import SFTTrainer
from datasets import load_dataset

model_name = "../../llama/llama-2-7b" 
new_model = "output"
device_map = {"": 0}

dataset = load_dataset("Anthropic/hh-rlhf", split="train")

base_model = AutoModelForCausalLM.from_pretrained( model_name,
low_cpu_mem_usage=True,
return_dict=True,
torch_dtype=torch.float16,
device_map=device_map,
)

args = TrainingArguments(output_dir="./output")

trainer = SFTTrainer(
    model = base_model,
    args = args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
)

trainer.train()