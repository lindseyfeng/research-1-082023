import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    Adafactor,
    HfArgumentParser,
    pipeline
)
from statistics import mean 
import re
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed, PreTrainedModelWrapper, create_reference_model
from trl.core import LengthSampler
import wandb
wandb.init(settings=wandb.Settings(_service_wait=60))


# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

tqdm.pandas()

model_dir = "./checkpoints/checkpoint-1000"
rm_tokenizer = AutoTokenizer.from_pretrained("weqweasdas/hh_rlhf_rm_open_llama_13b")
seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"
  
rm_pipe = pipeline(
      "sentiment-analysis",
      model="weqweasdas/hh_rlhf_rm_open_llama_13b",
      device=device,
      tokenizer=rm_tokenizer,
      model_kwargs={"torch_dtype": torch.bfloat16}
  )

pipe_kwargs = {
      "return_all_scores": True,
      "function_to_apply": "none",
      "batch_size": 1
    }

# parser = HfArgumentParser(ScriptArguments)
# script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

set_seed(seed)

def remove_tags(text):
    """
    Removes the substring "###Assistant" and any substring starting with "###human" in the middle of the text.
    :param text: str, the input text
    :return: str, the text with specified substrings removed
    """
    # Remove "###Assistant"
    text = text.replace("###Assistant: ", "")

    # Remove any substring starting with "###human"
    text = re.sub(r'###Human.*$', '', text, flags=re.IGNORECASE)

    return text.strip()

# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
        tokenizer, dataset_name, input_min_text_length=2, input_max_text_length=8
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    train_dataset = load_dataset(dataset_name, split="train")
    original_columns = train_dataset.column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["chosen"]:
            # dialogues = text.split("Assistant:")
            # question = [part.split("Human:")[1].strip() for part in dialogues if "Human:" in part][0]
            start_index = question.find("Human")
            end_index = question.find("Assistant")
            question = question[start_index+6:end_index].strip()
            query = question
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])
        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    # ds = ds.filter(lambda x: len(x["input_ids"]) < script_args.max_length, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# config = PPOConfig(
#     model_name=script_args.model_name,
#     learning_rate=script_args.learning_rate,
#     log_with=script_args.log_with,
#     batch_size=script_args.batch_size,
#     mini_batch_size=script_args.mini_batch_size,
#     gradient_accumulation_steps=script_args.gradient_accumulation_steps,
#     optimize_cuda_cache=True,
#     early_stopping=script_args.early_stopping,
#     target_kl=script_args.target_kl,
#     ppo_epochs=script_args.ppo_epochs,
#     seed=script_args.seed,
# )

config = PPOConfig(
    steps = 2048,
    learning_rate=5e-6,
    init_kl_coef = 0.01,
    log_with="wandb",
    ppo_epochs= 4,
    batch_size = 16,
    gradient_accumulation_steps = 4, 
    kl_penalty = 'abs'
    )
  

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
rw_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True
}

tokenizer = LlamaTokenizer.from_pretrained(model_dir)
if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer, "Anthropic/hh-rlhf")

# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_dir,
    device_map={"": current_device},
    peft_config=lora_config,
)
print(model.training)
base_dir = "../../llama/llama-2-7b"
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_dir)
wrapped_model = PreTrainedModelWrapper(ref_model)
reference_model = create_reference_model(wrapped_model)


optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model = reference_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug


# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 0.9,
    "temperature": 2.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": -1,
    "max_new_tokens": 200,
}
output_min_length = 100
output_max_length = 200
output_length_sampler = LengthSampler(output_min_length, output_max_length)
save_freq = 200
output_dir= "./fllama_ppo"
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    question_tensors = batch["input_ids"]

    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    # Compute sentiment score
    # response = [remove_tags(r) for r in batch["response"]]
    texts = ["###Human: " + q +" ###Assistant: "+ r for q, r in zip(batch["query"], batch["response"])]
    # print(texts)
    # response_tensors = [torch.tensor(tokenizer.encode(r)) for r in response]
    pipe_outputs = rm_pipe(texts, **pipe_kwargs)
    tensor_rewards = [torch.tensor(output[0]["score"], dtype=torch.float32) for output in pipe_outputs]
    print(torch.mean(torch.stack(tensor_rewards), dim=0))
    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, tensor_rewards)
    ppo_trainer.log_stats(stats, batch, tensor_rewards)

 
    if save_freq and epoch and epoch % save_freq == 0:
        ppo_trainer.save_pretrained(output_dir + f"step_{epoch}")


        