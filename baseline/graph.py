import matplotlib.pyplot as plt

# Data
categories = ["no", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]
top_25 = [8.102625, 5.2119375, 4.50446875, 4.734765625, 4.61459375, 4.32278125, 4.51603125, 4.627625, 4.4284375, 4.4539375, 4.76953125, 4.408703125, 4.433078125, 4.40521875, 4.6249375]
mid_25 = [5.4725, 5.30609375, 5.08921875, 4.76203125, 4.92265625, 4.565320313, 5.03859375, 5.1244375, 4.81284375, 4.7054375, 4.987875, 4.62896875, 4.8318125, 4.929433594, 5.05496875]
bottom_25 = [1.963933105, 5.223796875, 4.972546875, 4.606382813, 4.508257813, 4.486851563, 4.508820313, 4.597007813, 4.484414063, 4.398476563, 4.8601875, 4.318726563, 4.268945313, 4.598851563, 4.836882813]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(categories[:-1], top_25, label="Top 25%", marker='o')
plt.plot(categories[:-1], mid_25, label="Mid 25%", marker='s')
plt.plot(categories[:-1], bottom_25, label="Bottom 25%", marker='^')

# Adding titles and labels
plt.xlabel("prompts")
plt.ylabel("rewards")
plt.xticks(rotation=45)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
