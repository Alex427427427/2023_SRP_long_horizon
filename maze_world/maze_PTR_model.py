import torch.nn as nn
import torch.nn.functional as F
import torch.optim

# create a preference learning model
class MazePTRModel(nn.Module):
    def __init__(self):
        super(MazePTRModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        # apply the linear layers to each state in the state pair
        #print(x[:,0].shape)
        x1 = self.fc(x[:, 0])
        #x1 = torch.exp(x1)
        t_left_1 = x1[:, 0].unsqueeze(1)
        x2 = self.fc(x[:, 1])
        #x2 = torch.exp(x2)
        t_left_2 = x2[:, 0].unsqueeze(1)
        return self.sigmoid(t_left_1 - t_left_2)
    
    def predict_time_to_goal(self, x):
        t = self.fc(x)
        #t = torch.exp(t)
        t = t[:, 0].unsqueeze(1)
        return t
