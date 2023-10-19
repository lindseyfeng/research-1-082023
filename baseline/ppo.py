from dataclasses import dataclass, field
from typing import Optional

import sys
sys.path.append('../')  # Append the parent directory to sys.path
import torch
import numpy as np

from accelerate import Accelerator
from datasets import load_dataset
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
#t5
from transformers import T5TokenizerFast

#bert
from transformers import BertModel, AutoTokenizer, DataCollatorForSeq2Seq
from bert.main import ModelWithTemperature, predict_scaled_sentiment, SentimentModel

tqdm.pandas()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

save_freq = 1000
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

#reward_model_name
bert_model = BertModel.from_pretrained('bert-base-uncased')
SentimentModel = SentimentModel(bert_model, 256, 1, 2, True, 0.25)
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
SentimentModel.load_state_dict(torch.load('model.pt', map_location=device))
scaled_model = ModelWithTemperature(SentimentModel)
scaled_model.load_state_dict(torch.load('model_with_temperature.pth', map_location=device))
best_temperature = scaled_model.temperature.item()


dataset_name = "imdb"

config = PPOConfig(
    steps=10000,
    ratio_threshold = 30
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
    ds = load_dataset(dataset_name, split="train")
    original_columns = ds.column_names
    num_proc = 24

    processed_dataset = ds.map(
    lambda examples: {"text": prefix + examples["text"]})


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



ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
)
print(ppo_trainer)

# We then build the sentiment analysis pipeline using our reward model, passing the
# model name and the sentiment analysis pipeline arguments. Let's also make sure to
# set the device to the same device as the PPOTrainer.
# device = ppo_trainer.accelerator.device
# if ppo_trainer.accelerator.num_processes == 1:
#     device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug
# sentiment_pipe = pipeline(
#     "sentiment-analysis",
#     model=reward_model_name,
#     device_map={"": current_device},
#     model_kwargs={"load_in_8bit": True},
#     tokenizer=tokenizer,
#     return_token_type_ids=False,
# )


# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.

generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
}
output_min_length = 48
output_max_length = 50
output_length_sampler = LengthSampler(output_min_length, output_max_length)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if epoch >= config.total_ppo_epochs:
        break
    print("epoch: {}".format(epoch))

    question_tensors = batch["input_ids"]

    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    print(response_tensors)
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Compute reward score (using the sentiment analysis pipeline)
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    rewards_list = [predict_scaled_sentiment(scaled_model, bert_tokenizer, output_text, best_temperature)for output_text in texts]
    rewards = [torch.tensor(reward) for reward in rewards_list]

    print(texts)
    print(rewards)
    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    if save_freq and epoch and epoch % save_freq == 0:
        ppo_trainer.save_pretrained(output_dir + f"step_{epoch}")

ppo_trainer.save_pretrained("./t5_imdb_ppo_complete")