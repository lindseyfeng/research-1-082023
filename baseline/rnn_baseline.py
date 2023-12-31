import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import zipfile
import os


def read_data(data_file='NYC_taxi_OD.npy'):
    # Load .npy file
    data = np.load(data_file)
    X = data.reshape([-1, 69, 69])
    # normalize
    X = (X - np.mean(X)) / np.std(X)
    return X

# Load the data from the .npy file
data = read_data()
print(data.shape)

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
def generate_data(data, in_length, predict_length):
    # Assuming data is a sequence of 69x69 matrices
    data_length = data.shape[0]
    X, Y = [], []
    for i in range(data_length - in_length - predict_length):
        X.append(data[i:i + in_length].reshape(in_length, -1))  # Flatten each matrix
        Y.append(data[i + in_length:i + in_length + predict_length].reshape(predict_length, -1))
    return np.array(X), np.array(Y)

# Generate sequences
in_length = 96
predict_length = 48
X, y = generate_data(data, in_length, predict_length)
print(X.shape)
print(y.shape)
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
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        out, hidden = self.rnn(x)
        return hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, output_length):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.output_length = output_length

        # Initialize layers
        self.rnn = nn.RNN(hidden_size, hidden_size,num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)


    def forward(self, hidden):
        output = torch.zeros((hidden.size(0), self.output_length, self.output_size))
        inp = torch.zeros((hidden.size(0), self.hidden_size))  # Initial input
        inp = inp.unsqueeze(1)
        inp = inp.repeat(1, 2, 1)
        hidden = hidden.repeat(1, 2, 1)
        for t in range(self.output_length):
            out, hidden = self.rnn(inp, hidden)
            out = self.linear(out)
            output[:, t, :] = out[0]

        return output


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        hidden = self.encoder(x)
        output = self.decoder(hidden)
        return output

# Model initialization
input_size = 69 * 69
hidden_size = 64
num_layers = 2
output_size = 69 * 69

encoder = EncoderRNN(input_size, hidden_size, num_layers)
decoder = DecoderRNN(hidden_size, output_size, num_layers, output_length=48)
model = Seq2Seq(encoder, decoder)

# Loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    for i in range(len(X_train)):
        seq = X_train[i].unsqueeze(0)
        labels = y_train[i].unsqueeze(0)
        optimizer.zero_grad()
        y_pred = model(seq)
        single_loss = loss_function(y_pred[0], labels[0])
        single_loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print(f'Epoch {epoch}, Train Loss: {single_loss.item()}')


    train_loss = single_loss.item()
    val_loss = evaluate_model(model, X_val, y_val)

    print(f'Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}')

test_loss = evaluate_model(model, X_test, y_test)
print(f'Test Loss: {test_loss}')