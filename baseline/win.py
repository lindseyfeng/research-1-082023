## This code computes completions for questions in a json file using Claude-v2 model from Bedrock
## Follow instructions here for setup before running this code:
## https://quip-amazon.com/Lvo3A3JhGftQ/Using-Claude-from-AWS-Bedrock-Endpoint

import json
import argparse
import os 
import datetime
import time
from tqdm import tqdm
import numpy as np
import openai
import random
import re

random.seed(42)

def extract_question_and_answer_with_re(text):
    # This regex pattern looks for the content after "###human: " and before " ###assistant:",
    # capturing both the question and the answer as separate groups without assumptions on the "###assistant:" content.
    pattern = r"###human: (.*?) ###assistant:(.*)"
    
    # Search for the pattern in the given text
    match = re.search(pattern, text, re.DOTALL)
    
    # If a match is found, return the captured groups (the question and the answer),
    # otherwise return None or an appropriate message.
    if match:
        question = match.group(1).strip()
        answer = match.group(2).strip()
        return question, answer
    else:
        return "No question and answer found.", ""

def call_gpt(data, args):
    openai.api_key = args.api_key
    response = openai.Completion.create(
        model="gpt-4", # Specify the model
        prompt="What is the capital of France?", # Your prompt to the model
        temperature=0, # Controls the randomness. Closer to 1 means more creative.
        max_tokens=100, # Maximum length of the output
        top_p=1, # Controls diversity via nucleus sampling: 0.5 means half of all likelihood-weighted options are considered.
        frequency_penalty=0, # Decreases the likelihood of repeating the same line verbatim.
        presence_penalty=0 # Decreases the likelihood of repeating the same topic.
    )
    return response


def pairwise_eval(args, data_path="../test_data/test_all.jsonl"):
    ties, wins, loses = 0,0,0
    with open(args.input_file1, 'r') as file:
        output1 = json.load(file)
    with open(args.input_file2, 'r') as file:
        output2 = json.load(file)
    
    # data = data[:730]

    # try:
    #     saved_idx = list(np.load(args.saved_idx).astype(int))
    # except:
    #     saved_idx = []    
    # annotated = []
    # for i in range(len(outputs1)):

        # if i in saved_idx:
        #     continue


    random_numbers = [random.randint(0, 999) for _ in range(100)]
    for i in random_numbers:
        results = []
        
        text1 = output1[i]
        text2 = output2[i]
        instruction1, ans1 = extract_question_and_answer_with_re(text1)
        instruction2, ans2 = extract_question_and_answer_with_re(text2)
        if instruction1 != insturction2:
            print(instruction1)
            print(insturction2)
            continue

        # Analyze with prompt 1 first
        with open("summarize_prompt.txt", "r") as f:
            text = f.read()

        text = text.replace("||instruction||", instruction1)
        text = text.replace("||output_1||", ans1)
        text = text.replace("||output_2||", ans2)

        completion = call_gpt(text, args)
        

        print(text, completion)

        if completion is None:
            result = -10
        elif "A" in completion[-5:]:
            result = 0
        elif "B" in completion[-5:]:
            result = 1
        else:
            result = -10
        results.append(result)

        # Analyze with prompt 2 first
        with open("basic_prompt.txt", "r") as f:
            text = f.read()
    
        text = text.replace("||instruction||", instruction)
        text = text.replace("||output_1||", output2)
        text = text.replace("||output_2||", output1)
        print(text)

        completion = call_gpt(text, args)

        print(text, completion)

        if completion is None:
            result = -10
        elif "A" in completion[-5:]:
            result = 1
        elif "B" in completion[-5:]:
            result = 0
        else:
            result = -10
        results.append(result)
        if sum(results) == 1:
            print("TIE")
            ties += 1
        elif sum(results) == 0:
            wins += 1
        elif sum(results) == 2:
            loses += 1
        annotated.append({"instruction": instruction,
                            "output1": output1,
                            "output2": output2,
                            "results": results})

        print("--------------------------------------------")
        print(f"{i} / {len(outputs1)}, ties: {ties}, wins: {wins}, loses: {loses}")
        print("--------------------------------------------")
        
        if -10 not in results:
            saved_idx.append(i)
            np.save(args.saved_idx, saved_idx)
                
            with jsonlines.open(f"{args.output_file}.jsonl", mode='a') as writer:
                writer.write_all([annotated[-1]])

    


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--input_file1", default="base_vicuna_ppo_response1000.json")
    parser.add_argument("--input_file2", default="dpo_vicuna_response1000.json")
    parser.add_argument("--output_file", default="./gpt_eval")
    parser.add_argument("--saved_idx")
    parser.add_argument("--max_tokens_to_sample", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=250)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--max_threads", type=int, default=2)
    args = parser.parse_args()

    pairwise_eval(args)