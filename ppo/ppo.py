import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#import gym
#from IPython import display

# Define the neural network architecture for the policy
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, state):
        x = self.fc(state)
        return torch.softmax(x, dim=-1)

# PPO agent class
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon_clip):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip

    def compute_advantage(self, rewards, values, dones):
        advantages = np.zeros((len(rewards), len(values[0])), dtype=np.float32)
        last_advantage = 0
        last_value = 0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * last_value * mask - values[t]
            last_advantage = delta + self.gamma * self.epsilon_clip * last_advantage * mask
            advantages[t], advantages[t] = last_advantage
            last_value = values[t]

        return advantages

    def update_policy(self, states, actions, old_probs, advantages, returns):
        #states = torch.FloatTensor(states)
        #actions = torch.LongTensor(actions)
        #old_probs = torch.FloatTensor(old_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        # Calculate new probabilities and ratios
        new_probs = self.policy(states)
        ratio = new_probs / old_probs
        print(new_probs.shape)
        print(old_probs.shape)
        print(ratio.shape)
        print(advantages.shape)

        # PPO loss function
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        print(returns.shape)
        print(self.policy(states).shape)
        # Value function loss
        value_loss = 0.5 * nn.MSELoss(returns, self.policy(states))

        # Total loss
        loss = policy_loss + value_loss

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()