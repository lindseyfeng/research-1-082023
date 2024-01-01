import json
import numpy as np
import matplotlib.pyplot as plt

# Load data from the JSON file
f = 'noprompt_results.json'
with open(f, 'r') as file:
    data = json.load(file)

# Extract scores from the JSON data
scores = data['rewards']

# Create a histogram of the scores
plt.hist(scores, bins=20, density=True, alpha=0.6, color='b', edgecolor='black')

# Calculate mean and standard deviation for a normal distribution
mean = np.mean(scores)
std_dev = np.std(scores)

# Generate a range of values for the normal distribution curve
x = np.linspace(min(scores), max(scores), 100)
y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev)**2)

# Plot the normal distribution curve
plt.plot(x, y, color='red', linewidth=2)

# Add labels and title
plt.xlabel('Score')
plt.ylabel('Percentage')
plt.title('Score Distribution for {}'.format(f))

# Show the plot
plt.show()
