# create a class for the environment

import numpy as np
import matplotlib.pyplot as plt
import torch
import os

import cv2

import gymnasium
from gymnasium import spaces

import torch.nn as nn
import torch.nn.functional as F
import torch.optim


image_path = "../mazes/maze_procedural_1.png"
model_path = "models/40_model.pt"


# all map representations follow classical cartesian coordinates, with the origin at the bottom left corner of the map,
# the x axis pointing to the right, and the y axis pointing upwards.
# the x cooresponding to the first index to arrays and the y corresponding ot the second index to arrays.

# create a function to take in png and convert into a numpy array
# takes in a png filename
# returns a numpy array
def png_to_occ_map(png_file):
    # read in the png file
    img = plt.imread(png_file)
    # convert to grayscale
    img = np.mean(img,axis=2)
    # convert to binary
    img = (img < 0.3).astype(np.float32) # everything below a certain darkness is an obstacle.
    return img

# extract the location of the goal
def find_goal_location(image_path):
    # Open the image
    img = plt.imread(image_path)

    # Get the width and height of the image
    height, width, channels = img.shape

    # Iterate through each pixel to find the red pixel
    for row in range(height):
        for col in range(width):
            # Get the RGB values of the pixel
            rgb = img[row, col]
            red = rgb[0]
            green = rgb[1]
            blue = rgb[2]
            # Check if it's a red pixel (adjust the threshold based on your image)
            if red > 0.7 and green < 0.3 and blue < 0.3:
                return (row, col)






# create a preference learning model
class MazePTRModel(nn.Module):
    def __init__(self, maze_size=40):
        super(MazePTRModel, self).__init__()
        
        # positional encoding
        self.feature_dimension = maze_size
        self.max_spatial_period = self.feature_dimension
        even_i = torch.arange(0, self.feature_dimension, 2).float()   # even indices starting at 0
        odd_i = torch.arange(1, self.feature_dimension, 2).float()    # odd indices starting at 1
        denominator = torch.pow(self.max_spatial_period, even_i / self.feature_dimension)
        positions = torch.arange(self.max_spatial_period, dtype=torch.float).reshape(self.max_spatial_period, 1)
        even_PE = torch.sin(positions / denominator)
        odd_PE =  torch.cos(positions / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        self.pe = torch.flatten(stacked, start_dim=1, end_dim=2)

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

    #def pe(self, x):
        #return self.final_PE[x]
    
    def forward(self, x):
        x = x.to(torch.int)
        # take the position encoding of the state
        # create an input tensor from the final position encoding table, where each row is the position encoding of a state
        aug_state_1 = torch.cat((self.pe[x[:, 0, 0]], self.pe[x[:, 0, 1]]), dim=1)
        aug_state_2 = torch.cat((self.pe[x[:, 1, 0]], self.pe[x[:, 1, 1]]), dim=1)

        x1 = self.fc(aug_state_1)
        t_left_1 = x1[:, 0].unsqueeze(1)
        x2 = self.fc(aug_state_2)
        t_left_2 = x2[:, 0].unsqueeze(1)
        return self.sigmoid(t_left_1 - t_left_2)
    
    def predict_time_to_goal(self, x):
        x = x.to(torch.int)
        aug_state = torch.cat((self.pe[x[:, 0]], self.pe[x[:, 1]]), dim=1)
        t = self.fc(aug_state)
        t = t[:, 0].unsqueeze(1)
        return t


# a reinforcement learning environment, a 2D maze. 
class MazeGym(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    # note the array is flipped when plotted
    def __init__(self,mode="human", sparse=True,model=None, move_penalty=0.05, goal_reward=10.0, collision_penalty=0.5, lifespan=400):
        super(MazeGym, self).__init__()
        self.mode = mode

        # create an occupancy map
        self.occ_map = png_to_occ_map(image_path)
        self.img = plt.imread(image_path)
        self.collision_penalty = collision_penalty
        self.Nx = self.occ_map.shape[0]
        self.Ny = self.occ_map.shape[1]

        # goal location and reward
        goal_tuple = find_goal_location(image_path)
        self.gx = goal_tuple[0] 
        self.gy = goal_tuple[1]
        self.goal_reward = goal_reward

        # truncation
        self.lifespan = lifespan
        self.age = 0

        # actions: up down left right
        self.action_look_up = np.array([[0,1],[0,-1],[-1,0],[1,0]]) # up down left right stay
        self.action_space = spaces.Discrete(len(self.action_look_up))
        self.move_penalty = move_penalty

        # states: x y
        # positional encoding
        self.feature_dimension = self.Nx
        self.max_spatial_period = self.Nx
        even_i = torch.arange(0, self.feature_dimension, 2).float()   # even indices starting at 0
        odd_i = torch.arange(1, self.feature_dimension, 2).float()    # odd indices starting at 1
        denominator = torch.pow(self.max_spatial_period, even_i / self.feature_dimension)
        positions = torch.arange(self.max_spatial_period, dtype=torch.float).reshape(self.max_spatial_period, 1)
        even_PE = torch.sin(positions / denominator)
        odd_PE =  torch.cos(positions / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        self.pe = torch.flatten(stacked, start_dim=1, end_dim=2)
        #self.observation_space = spaces.Box(low=np.array([0,0]), high=np.array([self.Nx-1,self.Ny-1]), dtype=np.int64)
        self.observation_space = spaces.Box(low=np.zeros(2*len(self.pe[0])), high=np.ones(2*len(self.pe[0])), dtype=np.float64)
        # randomly select a free state
        self.state = self.free_state_search()

        # initialise reward landscape
        self.reward_landscape = np.zeros((self.Nx, self.Ny))
        
        # fill out reward landscape
        self.sparse = sparse
        if sparse or model is None:
            self.reward_landscape[self.gx, self.gy] = goal_reward
        else:
            x = np.arange(self.Nx) # create x coord array
            y = np.arange(self.Ny) # create y coord array
            yy,xx = np.meshgrid(x,y) # create matrices of x and y coords, as separate matrices, that can be served as input to 
            # some multidimensional function
            self.model = MazePTRModel(maze_size=self.Nx)
            self.model.load_state_dict(torch.load(model_path))
            # create a tensor of the grid
            grid = torch.tensor(np.stack([xx.flatten(), yy.flatten()], axis=1), dtype=torch.float)
            # get the predictions
            time_proximity_to_goal = self.model.predict_time_to_goal(grid).detach().numpy()
            # reshape the predictions
            time_proximity_to_goal = time_proximity_to_goal.reshape(self.Nx, self.Ny)
            #rewards = np.exp(-1*time_proximity_to_goal)
            time_proximity_to_goal = (time_proximity_to_goal - np.min(time_proximity_to_goal))/(np.max(time_proximity_to_goal) - np.min(time_proximity_to_goal))
            rewards = 1 - time_proximity_to_goal
            self.times = time_proximity_to_goal
            self.reward_landscape = rewards
            self.reward_landscape[self.gx, self.gy] = goal_reward
        
    # apply a gaussian contraction on the reward landscape
    def shrink_reward(self, shrink_scaling=1, shrink_order=1):
        self.reward_landscape *= np.exp(-shrink_scaling*(self.times)**(2*shrink_order))

    # returns a random state free from obstacle from the entire map
    def free_state_search(self):
        while True:
            x0 = np.random.randint(self.Nx)
            y0 = np.random.randint(self.Ny)
            #if the location is not an obstacle, is not the goal, then break
            obs = self.occ_map[x0,y0]
            goal = (x0 == self.gx) and (y0 == self.gy)
            if (obs == 0) and (not goal):
                break
        return np.array([x0,y0])
        
    def reset(self, seed=None, options=None):
        info = {}
        self.age = 0
        self.state = self.free_state_search()
        state_index = np.round(self.state).astype(int)
        observation = np.concatenate((self.pe[state_index[0]], self.pe[state_index[1]]), axis=0)
        return np.copy(observation), info

    # apply action and update the state
    def movement_control(self,X,u):
        collision = False
        x = X[0] + u[0]
        y = X[1] + u[1]
        # if the new state is at an obstacle or out of the map, stay at the same state
        if x < 0 or x > self.Nx-1 or y < 0 or y > self.Ny-1 or self.occ_map[x,y] == 1:
            collision = True # a collision has happened
            return X, collision
        else:
            return np.array([x,y]), collision
    
    # a step of interaction
        # takes in action index
        # returns new state, reward, terminated
    def step(self,idx):
        terminated = False
        truncated = False
        info = {}

        # select action
        u = self.action_look_up[idx,:]
        # calculate next state and whether a collision has occured
        new_state, collision = self.movement_control(self.state,u)
        # overwrite the state
        self.state = np.copy(new_state)

        # find out if the agent has reached the goal
        if self.state[0] == self.gx and self.state[1] == self.gy:
            terminated = True

        # compute reward
        reward = self.reward_landscape[new_state[0],new_state[1]]
        # subtract movement penalty if the agent has not stayed still
        if idx != 4:
            reward = reward - self.move_penalty
        # subtract collision penalty
        if collision:
            reward = reward - self.collision_penalty

        self.age += 1
        if self.age >= self.lifespan:
            terminated = True
            truncated = True

        new_state_index = np.round(new_state).astype(int)
        new_observation = np.concatenate((self.pe[new_state_index[0]], self.pe[new_state_index[1]]), axis=0)

        return new_observation, reward, terminated, truncated, info
    
    def render(self, mode="human"):
        self.img[self.state[0],self.state[1],:] = [0,0,1]
        upsized = cv2.resize(self.img, dsize=(500, 500), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('maze', upsized)
        cv2.waitKey(100)
        self.img[self.state[0],self.state[1],:] = [1,1,1]

    def close(self):
        cv2.destroyAllWindows()

    