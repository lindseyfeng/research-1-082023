from dataclasses import dataclass, field
from typing import Optional

import sys
sys.path.append('../')  # Append the parent directory to sys.path
import torch
import numpy as np

from accelerate import Accelerator
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
#t5
from transformers import T5TokenizerFast
#log
import wandb

wandb.init()
tqdm.pandas()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

save_freq = 10
output_dir = "./t5_imdb_ppo"

prefix = "please complete the following: "

class LengthSampler1:
    """
    Samples a length
    """

    def __init__(self, min_value, max_value):
        self.values = list(range(min_value, max_value))

    def __call__(self):
        return np.random.choice(self.values)

#model_name, tokenizer_name
saved_directory = "./t5_imdb"
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(saved_directory)
tokenizer = T5TokenizerFast.from_pretrained(saved_directory)


dataset_name = "imdb"

config = PPOConfig(
    learning_rate=1.44e-5,
    init_kl_coef = 0.05,
    log_with="wandb",
    ppo_epochs= 1,
    mini_batch_size = 32,
    batch_size = 64,
    )

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True,
}


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
    tokenizer,
    dataset_name="imdb",
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
    
    # load imdb with datasets
    original_DS = load_dataset(dataset_name)
    num_proc = 24

    processed_dataset = original_DS.map(
    lambda examples: {"text": prefix + examples["text"]})["train"]

    def prepare_dataset(examples):
        length = LengthSampler1(80, 90)
        split_ids = [length() for _ in range(len(examples["text"]))]
        query = examples["text"]
        token_ids = tokenizer(examples["text"], truncation=True, max_length=120 ,padding='max_length',)
        input_ids = [ids[:idx]+[tokenizer.eos_token_id] for idx, ids in zip(split_ids, token_ids["input_ids"])]
        return {"query": query, "input_ids": input_ids}

    ds = processed_dataset.map(
        prepare_dataset,
        batched=True,
        remove_columns=["text", "label"]
    )
    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer)
print(dataset)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index
ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(saved_directory)

tokenizer.pad_token = tokenizer.eos_token

ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
)
print(ppo_trainer)

device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)


# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
output_min_length = 48
output_max_length = 100
output_length_sampler = LengthSampler(output_min_length, output_max_length)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if epoch >= config.total_ppo_epochs:
        break
    print("epoch: {}".format(epoch))

    question_tensors = batch["input_ids"]
    response_tensors = []
    for query in question_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # Compute reward score (using the sentiment analysis pipeline)
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
    # Run PPO step
    try:
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    except IndexError as e:
        print(f"Error occurred in ppo.py: {e}")
        # Handle the error here, for example:
        continue
    ppo_trainer.log_stats(stats, batch, rewards)

    if save_freq and epoch and epoch % save_freq == 0:
        ppo_trainer.save_pretrained(output_dir + f"step_{epoch}")

ppo_trainer.save_pretrained("./t5_imdb_ppo_complete")