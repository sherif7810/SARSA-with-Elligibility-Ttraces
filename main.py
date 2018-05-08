import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

env = gym.make('SpaceInvaders-v0')

env.render()


def wrap_state(state):
    """It wraps state in a tensor."""
    return torch.tensor(state).view(3, 210, 160).unsqueeze(0).float()


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


epsilon = 0.9
alpha = 0.622
ET_coef = torch.tensor(0.7)
gamma = torch.tensor(0.5)

model = DQN(env.action_space.n)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=alpha)


def epsilon_greedy(state):
    action = 0
    with torch.no_grad():
        if torch.rand(1)[0] > epsilon:
            action = torch.tensor(env.action_space.sample())
        else:
            action = torch.argmax(model(state), 1)
    return action


# SARSA with eligibility traces
for episode in range(0, 100):
    done = False
    G, reward = 0, 0

    E = torch.tensor(0.0)
    state = wrap_state(env.reset())
    action = epsilon_greedy(state)
    while done is not True:
        next_state, reward, done, info = env.step(action.item())
        next_state = wrap_state(next_state)
        next_action = epsilon_greedy(next_state)

        Q2 = model(next_state)[0, next_action]
        target = reward + gamma * Q2
        Q1 = model(state)[0, action]
        loss = criterion(target, Q1.detach())
        E += torch.tensor(1.)
        loss *= E
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        E *= gamma * ET_coef

        state, action = next_state, next_action

        G += reward
        env.render()

    episode_count = episode + 1
    if episode_count % 10 == 0:
        print("Episode {}: Total reward = {}.".format(episode_count, G))
