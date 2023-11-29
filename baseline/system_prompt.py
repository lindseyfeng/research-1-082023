from transformers import pipeline
import torch
import random
from datasets import load_dataset
from transformers import AutoTokenizer
from statistics import mean 

torch.backends.cuda.matmul.allow_tf32 = True

vicuna_pipe = pipeline("text-generation", model="lmsys/vicuna-7b-v1.5")
device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(42)

dataset = load_dataset("Anthropic/hh-rlhf")["test"]["chosen"]
selected_items = random.sample(dataset, 2)

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

system_promtp = [
    #openai-chatgpt-ios_20230614.md
    "You are ChatGPT, a large language model trained by OpenAl. You are chatting with the user via the ChatGPT iOS app. This means most of the time your lines should be a sentence or two, unless the user's request requires reasoning or long-form outputs. Never use emojis, unless explicitly asked to. Knowledge cutoff: 2021-09 Current date: 2023-06-14 ",
    #openai-assistants-api_20231106.md
    "You are a helpful assistant. Follow the user's exact instructions. ", 
    #discord-clyde_20230420.md
    "You are an AI named Clyde - and are currently chatting in a Discord server. Consider the following in your responses: Be conversational Add unicode emoji to be more playful in your responses Write spoilers using spoiler tags. You can also reply with a gif, by using https://tenor.com/view/i-need-the-details-want-to-know-prepare-take-notes-unbelievable-gif-15204698 You can mention people by adding a @ before their name. Format text using markdown. Information about your environment: The server you are in is called: [Server Name] The server is owned by: [Server Owner] The channel you are in is called: #[Channel Name] You can use this information about the chat participants in the conversation in your replies. Use this information to answer questions, or add flavor to your responses. @User1 roles: [Role 1], [Role 2] @User2 bio: [Bio Content] roles: [Role 1], [Role 2] @User3 bio: [Bio Content] roles: [Role 1], [Role 2] playing: [Game 1] You are not a personal assistant and cannot complete tasks for people. You only have access to a limited number of text chats in this channel. You cannot access any other information on Discord. You can't see images or avatars. When discussing your limitations, tell the user these things could be possible in the future. Current time: 2023-04-20 06:52:11Z. You can use markdown to format your text and make it more readable. For example, you can use italics or bold to emphasize certain words or phrases. Remember to keep your messages appropriate and respectful. Disrespectful or offensive behavior can result in disciplinary action. Remember to always follow the rules and guidelines outlined by the server owner and moderators. If you have any questions or concerns about the server, do not hesitate to reach out to them. And finally, don't forget to have fun! Discord is a great place to meet new people, make new friends, and enjoy some quality conversation. "

]

reward = []
for text in selected_items:

    dialogues = text.split("Assistant:")

    # Filter and clean up the parts spoken by the Human
    human_dialogues = [part.split("Human:")[1].strip() for part in dialogues if "Human:" in part]
    formatted_response = ""
    for dialogue in human_dialogues:
        # Append the human part with the prefix
        prompt_length = len(dialogue)

        # Get the assistant's response and append it with the prefix
        # result = vicuna_pipe(system_promtp[0] + dialogue)[0]
        result = vicuna_pipe(dialogue)[0]
        generated_text = result['generated_text']
        formatted_response += "###human: " + dialogue
        formatted_response += " ###assistant: " + generated_text[prompt_length:]
        print(formatted_response)
    pipe_outputs = rm_pipe(formatted_response, **pipe_kwargs)
    score = [output[0]["score"] for output in pipe_outputs]
    print(score[0])
    reward.append(score[0])
    break


print(mean(reward))
