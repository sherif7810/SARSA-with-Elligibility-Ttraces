import gym

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

env = gym.make('SpaceInvaders-v0')

env.render()


def wrap_state(state):
    """It wraps state in a Variable."""
    return Variable(torch.Tensor(state).view(3, 210, 160)).unsqueeze(0)


class DQN(nn.Module):
    """A NN from state to actions."""
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(22528, 256)
        self.fc5 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)


alpha, epsilon, ET_coef = (0.622, 0.9, 0.7)
gamma = torch.Tensor([0.5])
model = DQN(env.action_space.n)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=alpha)


def epsilon_greedy(state):
    action = 0
    Q = model(state)
    if torch.rand(1)[0] > epsilon:
        action = env.action_space.sample()
    else:
        action = Q.data.max(1)[1]
    return (action, Q)


# SARSA with eligibility traces
for episode in range(0, 100):
    done = False
    G, reward = 0, 0

    E = 0.0
    state1 = wrap_state(env.reset())
    action1, Q1 = epsilon_greedy(state1)
    while done is not True:
        state2, reward, done, info = env.step(action1)
        state2 = wrap_state(state2)
        action2, Q2 = epsilon_greedy(state2)

        Q2.data[0][action2] += reward
        Q2.data = Q2.data.mul(gamma)
        Q1 = Variable(Q1.data)
        loss = criterion(Q2, Q1)
        E += 1
        loss.data *= E
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state1, action1 = state2, action2
        E *= gamma * ET_coef

        G += reward
        env.render()

    episode_count = episode + 1
    if episode_count % 10 == 0:
        print("Episode {}: Total reward = {}.".format(episode_count, G))
