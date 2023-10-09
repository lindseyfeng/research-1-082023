
## Parameters to change

- positive_samples, random_positive_samples 
```python
positive_samples = [sample for sample in tokenized_datasets["train"] if sample['label'] == 1]
random_positive_samples = random.sample(positive_samples, 1000) #1k training sample
```
- train_dataloader
```python
train_dataloader = DataLoader(random_positive_samples, shuffle=True, batch_size=100, collate_fn=collate_fn) #100
```




## Experiments

[Google sheets](https://docs.google.com/spreadsheets/d/1YaA09TsEn-8NqBN8cu-utN-ViQBG7j2K4njdjFfppD4/edit?usp=sharing)


## Setup

We follow the training procedure addressed in the [RAFT paper](https://docs.google.com/spreadsheets/d/1YaA09TsEn-8NqBN8cu-utN-ViQBG7j2K4njdjFfppD4/edit?usp=sharing). 

Specifically, we first finetune a T5-large model (for computational consideration) for 1 epoch with 25k training samples from the IMDB dataset (see `baseline/finetune_t5.py`) 

Then, we train a reward model, specifically a bert (bert-base-uncased) classifier, also with 25k training samples from the IMDB dataset. (see `bert/main.py`)

We then combine these two models in `baseline/raft_baseline.py`. We first truncate the first 64 tokens of the reviews and pass these samples to the finetuned T5 for inference. We then collect the generated samples from T5 and pass them through BERT to get reward scores. We ranked these scores and pass in 20% of the samples with top score to our T5 model for further tuning. 

Note: The original paper use batch_size = 1048 and for computation consideration we are using batch_size = 100 in the code. these parameters can be changed for future experiments. 
