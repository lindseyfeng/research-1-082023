import json

# Load data from the JSON file
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to filter data based on reward percentiles
def filter_data_by_rewards(data):
    # Sort data by rewards
    combined_data = list(zip(data['rewards'], data['responses']))
    sorted_data = sorted(combined_data, key=lambda x: x[0], reverse=True)
    total_samples = len(sorted_data)

    # Calculate indices for reward percentiles
    top_25_index = total_samples // 4
    mid_25_index = total_samples // 2
    bottom_25_index = (total_samples * 3) // 4

    # Filter data
    top_25_percent = sorted_data[:top_25_index]
    mid_25_percent = sorted_data[top_25_index:mid_25_index]
    bottom_25_percent = sorted_data[bottom_25_index:]

    return top_25_percent, mid_25_percent, bottom_25_percent

def calculate_average_reward(reward_data):
    total_reward = sum(reward for reward, _ in reward_data)
    print("len(reward): {}".format(len(reward_data)))
    average_reward = total_reward / len(reward_data) if reward_data else 0
    return average_reward


# Path to your JSON file
file_path = 'prompt9_results.json'

# Load and filter the data
data = load_data(file_path)
top_25, mid_25, bottom_25 = filter_data_by_rewards(data)
average_reward_top_25 = calculate_average_reward(top_25)
print(top_25[0])
print(average_reward_top_25)
average_reward_mid_25 = calculate_average_reward(mid_25)
print(mid_25[0])
print(average_reward_mid_25)
average_reward_bottom_25 = calculate_average_reward(bottom_25)
print(bottom_25[-1])
print(average_reward_bottom_25)

