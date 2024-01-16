import torch.nn as nn
import torch.nn.functional as F
import torch.optim

# create a preference learning model
class MazePTRModel(nn.Module):
    def __init__(self):
        super(MazePTRModel, self).__init__()
        
        # positional encoding
        self.feature_dimension = 10
        self.max_spatial_period = 40
        even_i = torch.arange(0, self.feature_dimension, 2).float()   # even indices starting at 0
        odd_i = torch.arange(1, self.feature_dimension, 2).float()    # odd indices starting at 1
        denominator = torch.pow(self.max_spatial_period, even_i / self.feature_dimension)
        positions = torch.arange(self.max_spatial_period, dtype=torch.float).reshape(self.max_spatial_period, 1)
        even_PE = torch.sin(positions / denominator)
        odd_PE =  torch.cos(positions / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        self.final_PE = torch.flatten(stacked, start_dim=1, end_dim=2)

        # network
        self.fc = nn.Sequential(
            nn.Linear(2*self.feature_dimension, 800), # augmented input
            nn.LeakyReLU(),
            nn.Linear(800, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 1)
        )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()

    def pe(self, x):
        return self.final_PE[x]
    
    def forward(self, x):
        # apply the linear layers to each state in the state pair
        # take the position encoding of the state
        aug_state_1 = torch.cat([self.final_PE[a] for a in x[:, 0, 0]], dim=0)
        aug_state_2 = torch.cat([self.final_PE[a] for a in x[:, 0, 0]], dim=0)

        x1 = self.fc(x[:, 0])
        
        #x1 = torch.exp(x1)
        t_left_1 = x1[:, 0].unsqueeze(1)
        x2 = self.fc(x[:, 1])
        #x2 = torch.exp(x2)
        t_left_2 = x2[:, 0].unsqueeze(1)
        return self.sigmoid(t_left_1 - t_left_2)
    
    def predict_time_to_goal(self, x):
        x = torch.cat((x, torch.ones(x.shape[0], 1)), dim=1)
        t = self.fc(x)
        #t = torch.exp(t)
        t = t[:, 0].unsqueeze(1)
        return t


# create a preference learning model
class MazePTRModel2(nn.Module):
    def __init__(self):
        super(MazePTRModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 1600), # augmented input
            nn.LeakyReLU(),
            nn.Linear(1600, 800),
            nn.LeakyReLU(),
            nn.Linear(800, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 1)
        )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        # apply the linear layers to each state in the state pair
        #print(x[:,0].shape)
        # append a 1 along dimension 2
        x = torch.cat((x, torch.ones(x.shape[0], x.shape[1], 1)), dim=2)
        x1 = self.fc(x[:, 0])
        #x1 = torch.exp(x1)
        t_left_1 = x1[:, 0].unsqueeze(1)
        x2 = self.fc(x[:, 1])
        #x2 = torch.exp(x2)
        t_left_2 = x2[:, 0].unsqueeze(1)
        return self.sigmoid(t_left_1 - t_left_2)
    
    def predict_time_to_goal(self, x):
        x = torch.cat((x, torch.ones(x.shape[0], 1)), dim=1)
        t = self.fc(x)
        #t = torch.exp(t)
        t = t[:, 0].unsqueeze(1)
        return t

