import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# Load the data
data = np.load('/mnt/data/NYC_taxi_OD.npy')

def evaluate_model(model, X, y):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        total_loss = 0
        for i in range(len(X)):
            seq = X[i].unsqueeze(0)
            labels = y[i].unsqueeze(0)
            y_pred = model(seq)
            loss = loss_function(y_pred, labels)
            total_loss += loss.item()
        average_loss = total_loss / len(X)
    return average_loss

# Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 100
X, y = create_sequences(data, seq_length)

# First split: Separate out the training data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Second split: Split the remaining data into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(2/3), random_state=42)  # 2/3 of 30% = 20%

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.linear(out[:, -1, :])
        return out

input_size = data.shape[1]
hidden_size = 64
num_layers = 2
output_size = data.shape[1]

model = RNNModel(input_size, hidden_size, num_layers, output_size)

# Loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    for i in range(len(X_train)):
        seq = X_train[i].unsqueeze(0)
        labels = y_train[i].unsqueeze(0)

        optimizer.zero_grad()
        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    train_loss = single_loss.item()
    val_loss = evaluate_model(model, X_val, y_val)

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}')

test_loss = evaluate_model(model, X_test, y_test)
print(f'Test Loss: {test_loss}')