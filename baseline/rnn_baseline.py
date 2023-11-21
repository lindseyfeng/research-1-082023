import torch
import torch.nn as nn
import numpy as np
import scipy.io

# Load the .mat file
file_path = 'SAND_TM_Estimation_Data.mat'
mat = scipy.io.loadmat(file_path)

# Print the keys to see what data is available
print(mat.keys())

print("odnames shape:", odnames.shape, "type:", odnames.dtype)
print("A shape:", A.shape, "type:", A.dtype)
print("X shape:", X.shape, "type:", X.dtype)
print("edgenames shape:", edgenames.shape, "type:", edgenames.dtype)
# Convert data to PyTorch tensors
data = torch.tensor(data, dtype=torch.float32)

# Create sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append((seq, label))
    return sequences

sequences = create_sequences(data, sequence_length)
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

# Initialize the RNN model
input_size = matrix_size * matrix_size  # Flattened OD matrix
hidden_size = 50  # Number of features in the hidden state
num_layers = 1  # Number of stacked LSTM layers

model = RNNModel(input_size, hidden_size, num_layers)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for seq, labels in sequences:
        optimizer.zero_grad()
        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if epoch % 10 == 1:
        print(f'epoch: {epoch:3} loss: {single_loss.item():10.8f}')
