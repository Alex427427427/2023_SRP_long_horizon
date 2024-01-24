import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DeepQ(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DeepQ, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, action_dim),
        )
        # initialize weights as uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, state):
        x = self.fc(state)
        return x
    
class Agent():
    def __init__(self, state_dim=2, action_dim=5, lr=0.01, gamma=0.99, memory_capacity=100000):
        self.state_dim = state_dim
        
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.model_main = DeepQ(state_dim, action_dim)
        self.model_target = DeepQ(state_dim, action_dim)
        self.model_target.load_state_dict(self.model_main.state_dict())
        self.optimizer = optim.Adam(self.model_main.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.memory_capacity = memory_capacity
        #dummy_state_1 = np.zeros(self.state_dim)
        #dummy_state_2 = np.zeros(self.state_dim)
        #self.memory = [(dummy_state_1, 0, dummy_state_2, 0, 0)]*self.memory_capacity
        self.memory = []
        self.memory_counter = 0
        
    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float)
            state = state.unsqueeze(0)
            q_value = self.model_main(state)
            action = torch.argmax(q_value, dim=1).item()
            return action
        
    def add_to_memory(self, transition):
        # a tuple of ([xt, yt], at, [xt+1, yt+1], rt+1, done)
        # if the length of the memory is less than the capacity, append
        if len(self.memory) < self.memory_capacity:
            self.memory.append(transition)
        # if the length of the memory is equal to the capacity, replace. Then increment the counter and reset it to 0
        elif len(self.memory) == self.memory_capacity:
            self.memory[self.memory_counter] = transition
            self.memory_counter += 1
            if self.memory_counter == self.memory_capacity:
                self.memory_counter = 0

    def get_batch_from_memory(self, batch_size):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        # sample a batch of transitions from memory
        batch_indices = np.random.choice(len(self.memory), batch_size, replace=False) # sample indices without replacement
        batch = [self.memory[i] for i in batch_indices] # create the actual batch by list processing
        return batch

    def learn_step(self, batch):
        # unpack the batch
        state_batch = torch.tensor([transition[0] for transition in batch], dtype=torch.float) # bs x state_dim
        action_batch = torch.tensor([transition[1] for transition in batch], dtype=torch.long).unsqueeze(1) # bs x 1
        next_state_batch = torch.tensor([transition[2] for transition in batch], dtype=torch.float) # bs x state_dim
        reward_batch = torch.tensor([transition[3] for transition in batch], dtype=torch.float).unsqueeze(1) # bs x 1
        done_batch = torch.tensor([transition[4] for transition in batch], dtype=torch.float).unsqueeze(1) # bs x 1
        
        # compute the loss
        old_state_q_values = self.model_main(state_batch).gather(1, action_batch)
        next_state_q_values = self.model_target(next_state_batch).max(dim=1)[0].unsqueeze(1) # return type is a tuple, where the first is the tensor and the second is the indices
        old_state_target_q_values = reward_batch + self.gamma * next_state_q_values * (1 - done_batch)
        loss = self.loss(old_state_q_values, old_state_target_q_values)
        
        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.model_target.load_state_dict(self.model_main.state_dict())
