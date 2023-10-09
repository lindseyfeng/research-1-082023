
# Experiments

we sample 100 prompts and, just like the training process, we truncate the first 64 tokens and feed it to our tuned T5 model in `./t5_imdb_complete` for inference. Then, we calculate scores of different metrics to assess the quality of our generated text.

# Metrics 
- mean sentiment score 
- [Distinct 1, 2](https://github.com/neural-dialogue-metrics/Distinct-N/tree/main)
- [Diverse 1, 2](https://github.com/XinnuoXu/mmi_anti_pytorch)
- [Unique 1, 2]()
- [MSTTR](https://github.com/LCR-ADS-Lab/TAALED)


For more metrics, see [this paper](https://aclanthology.org/2021.gem-1.10.pdf)
