import torch
import torch.nn as nn
import numpy as np
import scipy.io
from statistics import mean
from sklearn.preprocessing import MinMaxScaler


# Load the .mat file
file_path = 'SAND_TM_Estimation_Data.mat'
mat = scipy.io.loadmat(file_path)

# Print the keys to see what data is available
print(mat.keys())
odnames = mat['odnames']
A = mat['A']
X = mat['X']
edgenames = mat['edgenames']
# print("odnames shape:", odnames.shape, "type:", odnames.dtype)
# print("A shape:", A.shape, "type:", A.dtype)
# print("X shape:", X.shape, "type:", X.dtype)
# print("edgenames shape:", edgenames.shape, "type:", edgenames.dtype)
# # Convert data to PyTorch tensors

# Normalize the X data
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# Function to create sequences
def create_sequences(input_data, seq_length):
    xs = []
    ys = []
    for i in range(len(input_data) - seq_length):
        x = input_data[i:(i + seq_length)]
        y = input_data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 5  # Number of time steps in each input sequence
X_seq, y_seq = create_sequences(X_scaled, seq_length)

# Convert to PyTorch tensors
X_seq = torch.tensor(X_seq, dtype=torch.float32)
y_seq = torch.tensor(y_seq, dtype=torch.float32)

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, h_n = self.rnn(x)
        out = self.linear(out[:, -1, :])
        return out

input_size = 121  
hidden_size = 50 
num_layers = 1  


model = RNNModel(input_size, hidden_size, num_layers)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Calculate the split index
split_idx = int(len(X_seq) * 0.9)

# Split the data into training and testing sets
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]


num_epochs = 10  

for epoch in range(num_epochs):
    for i in range(len(X_train)):
        print(len(X_train))
        optimizer.zero_grad()
        seq = X_train[i].unsqueeze(0)  
        labels = y_train[i].unsqueeze(0) 

        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()
        print(f'epoch: {epoch:3} loss: {single_loss.item():10.10f}')

loss = []
for i in range(len(X_test)):
    seq = X_test[i].unsqueeze(0)  
    labels = y_test[i].unsqueeze(0) 
    y_pred = model(seq)
    single_loss = loss_function(y_pred, labels)
    loss.append(single_loss.item())
print(mean(loss))

