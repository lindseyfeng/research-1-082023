import torch
from transformers import ( AutoModelForCausalLM, AutoTokenizer,
pipeline,
)
from transformers import TrainingArguments
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

lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
)

args = TrainingArguments(output_dir="./output", report_to="wandb",)

trainer = SFTTrainer(
    model = base_model,
    args = args,
    train_dataset=dataset,
    dataset_text_field="chosen",
    max_seq_length=512,
    dataset_batch_size = 8,
    peft_config=lora_config
)

trainer.train()
print("Saving last checkpoint of the model")
trainer.model.save_pretrained(output_dir+" final_checkpoint/"))