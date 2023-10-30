import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train_rm(rm, state_buffer, reward_buffer, bsz = 16, n_batch=16, sigma_mult=1):
    state_buffer = torch.Tensor(np.stack(state_buffer))
    reward_buffer = torch.Tensor(np.stack(reward_buffer))
    sigmas = torch.Tensor(reward_buffer.std(0)) * sigma_mult
    total_loss = 0
    total_acc = 0
    total = 0
    reward_scale = []
    for i in range(n_batch):
        idx = np.random.choice(state_buffer.shape[0], bsz)
        loss, acc, outs = rm.get_loss(state_buffer[idx], reward_buffer[idx], sigmas)
        if loss <= 0:
            continue
        reward_scale += outs
        rm.optimizer.zero_grad()
        loss.backward()
        rm.optimizer.step()

        total_loss += loss.item()
        total_acc += acc
        total += 1

    rm.mu, rm.sigma = np.array(reward_scale).mean(), np.array(reward_scale).std()

    return total_loss / (total+1e-5), total_acc / (total+1e-5)

class RewardModel(nn.Module):
    def __init__(self, lr, normalize=False):
        super().__init__()
        
        self.fc1 = nn.Linear(6, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        self.mu = 0
        self.sigma = 1
        self.normalize = normalize

        self.heirarchy = [0, 1] # Now we only have two metrics: sentiment and diversity

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.normalize:
            x = (x - self.mu) / self.sigma
        return x

    def get_loss(self, x, reward_signal, sigmas):
        sign = [-1, -1] # One sign for each metric: sentiment and diversity
        total_loss = 0
        total = 0
        correct = 0
        outs = []

        for i in range(x.shape[0]):
            for j in range(i):
                x_i = x[i]
                x_j = x[j]

                reward_i = self(x_i)
                if j == 0:
                    outs.append(reward_i.item())
                reward_j = self(x_j)

                reward_info_i = reward_signal[i]
                reward_info_j = reward_signal[j]

                # Level 1: sentiment_Score
                if reward_info_i[self.heirarchy[0]] * sign[0] > reward_info_j[self.heirarchy[0]] * sign[0] + sigmas[self.heirarchy[0]]:
                    loss = -1 * torch.log(torch.sigmoid(reward_i - reward_j))
                    if reward_i > reward_j:
                        correct += 1
                elif reward_info_j[self.heirarchy[0]] * sign[0] > reward_info_i[self.heirarchy[0]] * sign[0] + sigmas[self.heirarchy[0]]:
                    loss = -1 * torch.log(torch.sigmoid(reward_j - reward_i))
                    if reward_j > reward_i:
                        correct += 1
                # Level 2: diversity_Score
                elif reward_info_i[self.heirarchy[1]] * sign[1] > reward_info_j[self.heirarchy[1]] * sign[1] + sigmas[self.heirarchy[1]]:
                    loss = -1 * torch.log(torch.sigmoid(reward_i - reward_j))
                    if reward_i > reward_j:
                        correct += 1
                elif reward_info_j[self.heirarchy[1]] * sign[1] > reward_info_i[self.heirarchy[1]] * sign[1] + sigmas[self.heirarchy[1]]:
                    loss = -1 * torch.log(torch.sigmoid(reward_j - reward_i))
                    if reward_j > reward_i:
                        correct += 1
                else:
                    continue

                total += 1
                total_loss += loss

        return total_loss / (total + 1e-5), correct / (total + 1e-5), outs
