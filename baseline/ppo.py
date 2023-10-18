from dataclasses import dataclass, field
from typing import Optional

import sys
sys.path.append('../')  # Append the parent directory to sys.path
import torch

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
# @dataclass
# class ScriptArguments:
#     """
#     The name of the Casual LM model we wish to fine with PPO
#     """

#     # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
#     # models like gpt-neo* models are more suitable.
#     model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
#     tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
#     reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
#     log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
#     learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
#     output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
#     mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
#     batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
#     ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
#     gradient_accumulation_steps: Optional[int] = field(
#         default=4, metadata={"help": "the number of gradient accumulation steps"}
#     )
#     adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
#     early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
#     target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
#     reward_baseline: Optional[float] = field(
#         default=0.0,
#         metadata={"help": "a baseline value that is subtracted from the reward"},
#     )
#     batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
#     save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
#     output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
#     seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
#     steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
#     init_kl_coef: Optional[float] = field(
#         default=0.2,
#         metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
#     )

#     adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})


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
    steps=1024,
    init_kl_coef=0.05,
)

train_dataset = load_dataset(dataset_name)

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
    ds = load_dataset(dataset_name, data_dir="data/rl", split="train")
    original_columns = ds.column_names
    num_proc = 24

    def prepare_dataset(examples):
        length = LengthSampler(50, 60)
        split_ids = [length() for _ in range(len(examples["text"]))]
        token_ids = tokenizer(examples["text"], truncation=True, max_length=120 ,padding='max_length',)
        input_ids = [ids[:idx]+[tokenizer.eos_token_id] for idx, ids in zip(split_ids, token_ids["input_ids"])]
        label_ids = [ids[idx:] for idx, ids in zip(split_ids, token_ids["input_ids"])]
        return {"input_ids": input_ids, "labels": label_ids}

    ds = train_dataset.map(
        prepare_dataset,
        batched=True,
        remove_columns="text"
    )

    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer)


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
output_max_length = 48
output_length_sampler = LengthSampler(output_min_length, output_max_length)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if epoch >= config.total_ppo_epochs:
        break

    question_tensors = batch["input_ids"]

    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Compute reward score (using the sentiment analysis pipeline)
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    rewards = [predict_scaled_sentiment(scaled_model, bert_tokenizer, output_text, best_temperature)for output_text in texts]

    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    if save_freq and epoch and epoch % save_freq == 0:
        ppo_trainer.save_pretrained(output_dir + f"step_{epoch}")

ppo_trainer.save_pretrained("./t5_imdb_ppo_complete")