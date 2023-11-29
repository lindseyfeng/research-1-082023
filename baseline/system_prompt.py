from transformers import pipeline
import torch
import random
from datasets import load_dataset
from transformers import AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True

vicuna_pipe = pipeline("text-generation", model="lmsys/vicuna-7b-v1.5")
device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(42)

dataset = load_dataset("Anthropic/hh-rlhf")["test"]["chosen"]
selected_items = random.sample(dataset, 3000)

# Path to your finetuned model
model_path = './LMFlow/output_models/finetuned_vicuna'

# Load the model
rm_tokenizer = AutoTokenizer.from_pretrained("weqweasdas/hh_rlhf_rm_open_llama_3b")
  
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

# pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
# rewards = [output[0]["score"] for output in pipe_outputs]

reward = []
for text in selected_items:

    dialogues = text.split("Assistant:")

    # Filter and clean up the parts spoken by the Human
    string = ""
    human_dialogues = [part.split("Human:")[1].strip() for part in dialogues if "Human:" in part]

    for dialogue in human_dialogues:
        # Append the human part with the prefix
        prompt_length = len(dialogue)

        string += "###human: " + dialogue + "\n"

        # Get the assistant's response and append it with the prefix
        result = vicuna_pipe(dialogue)[0]
        generated_text = result['generated_text']
        formatted_response = "###human: " + dialogue
        formatted_response += "\n###assistant:" + generated_text[prompt_length:]
        print(formatted_response)

    reward.append(score)
