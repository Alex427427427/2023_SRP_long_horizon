import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

# Define the neural network architecture for the policy
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, state):
        x = torch.relu(self.fc(state))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

# PPO agent class
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon_clip):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip

    def compute_advantage(self, rewards, values, dones):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_advantage = 0
        last_value = 0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * last_value * mask - values[t]
            last_advantage = delta + self.gamma * self.epsilon_clip * last_advantage * mask
            advantages[t] = last_advantage
            last_value = values[t]

        return advantages

    def update_policy(self, states, actions, old_probs, advantages, returns):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_probs = torch.FloatTensor(old_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        # Calculate new probabilities and ratios
        new_probs = self.policy(states).gather(1, actions.unsqueeze(1))
        ratio = new_probs / old_probs

        # PPO loss function
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value function loss
        value_loss = 0.5 * nn.MSELoss()(returns, self.policy(states))

        # Total loss
        loss = policy_loss + value_loss

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Environment setup
env = gym.make('CartPole-v1')  # Replace with your 2D maze environment

# PPO parameters
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
lr = 0.001
gamma = 0.99
epsilon_clip = 0.1

ppo_agent = PPOAgent(state_dim, action_dim, lr, gamma, epsilon_clip)

# Training loop
num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    states, actions, rewards, old_probs, values = [], [], [], [], []

    while not done:
        # Collect data
        action_probs = ppo_agent.policy(torch.FloatTensor(state))
        action = torch.multinomial(action_probs, 1).item()
        value = ppo_agent.policy(torch.FloatTensor(state)).detach().numpy()
        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        old_probs.append(action_probs[action].item())
        values.append(value)

        state = next_state
        total_reward += reward

    # Compute returns and advantages
    returns = []
    advantages = ppo_agent.compute_advantage(rewards, values, done)

    for t in range(len(rewards)):
        Gt = np.sum([r * (ppo_agent.gamma ** i) for i, r in enumerate(rewards[t:])])
        returns.append(Gt)

    # Normalize advantages
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    # Update policy
    ppo_agent.update_policy(states, actions, old_probs, advantages, returns)

    # Print episode info
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# Close the environment
env.close()