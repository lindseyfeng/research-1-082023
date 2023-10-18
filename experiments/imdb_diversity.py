import sys
import random
sys.path.append('../')

from datasets import load_dataset, load_metric, Dataset  
from distinct_n.metrics import distinct_n_corpus_level

dataset = load_dataset("imdb")
train_test = dataset["train"]["text"]
test_test = dataset["test"]["text"]
unsupervised_test = dataset["unsupervised"]["text"]
distinct_1 = distinct_n_corpus_level(train_test,1)
distinct_2 = distinct_n_corpus_level(train_test,2)
print("train")
print("DISTINCT-1", distinct_1)
print("DISTINCT-2", distinct_2)

distinct_1 = distinct_n_corpus_level(test_test,1)
distinct_2 = distinct_n_corpus_level(test_test,2)
print("test")
print("DISTINCT-1", distinct_1)
print("DISTINCT-2", distinct_2)
distinct_1 = distinct_n_corpus_level(unsupervised_test,1)
distinct_2 = distinct_n_corpus_level(unsupervised_test,2)
print("unsupervised")
print("DISTINCT-1", distinct_1)
print("DISTINCT-2", distinct_2)