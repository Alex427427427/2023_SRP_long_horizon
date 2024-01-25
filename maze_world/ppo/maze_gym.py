# create a class for the environment

import numpy as np
import matplotlib.pyplot as plt
import torch
import os

import cv2

import gymnasium
from gymnasium import spaces


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
    img = (img < 0.1).astype(np.float32) # everything below a certain darkness is an obstacle.
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

image_path = "maze_procedural_1.png"

# a reinforcement learning environment, a 2D maze. 
class MazeGym(gymnasium.Env):
    # note the array is flipped when plotted
    def __init__(self,sparse=True,model=None, move_penalty=0.05, goal_reward=10.0, collision_penalty=0.5, lifespan=1000):
        super(MazeGym, self).__init__()
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
        self.action_look_up = np.array([[0,1],[0,-1],[-1,0],[1,0],[0,0]]) # up down left right stay
        self.action_space = spaces.Discrete(len(self.action_look_up))
        self.move_penalty = move_penalty

        # states: x y
        self.observation_space = spaces.Box(low=np.array([0,0]), high=np.array([self.Nx-1,self.Ny-1]), dtype=np.int64)
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
            self.model = model
            # create a tensor of the grid
            grid = torch.tensor(np.stack([xx.flatten(), yy.flatten()], axis=1), dtype=torch.float)
            # get the predictions
            time_proximity_to_goal = model.predict_time_to_goal(grid).detach().numpy()
            # reshape the predictions
            time_proximity_to_goal = time_proximity_to_goal.reshape(self.Nx, self.Ny)
            #rewards = np.exp(-1*time_proximity_to_goal)
            time_proximity_to_goal = (time_proximity_to_goal - np.min(time_proximity_to_goal))/(np.max(time_proximity_to_goal) - np.min(time_proximity_to_goal))
            rewards = 1 - time_proximity_to_goal
            self.times = time_proximity_to_goal
            self.reward_landscape = rewards
            #self.reward_landscape[self.gx, self.gy] = goal_reward
        
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
        return np.copy(self.state), info

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

        return new_state, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        self.img[self.state[0],self.state[1],:] = [0,0,1]
        upsized = cv2.resize(self.img, dsize=(500, 500), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('maze', upsized)
        cv2.waitKey(1)
        self.img[self.state[0],self.state[1],:] = [1,1,1]

    def close(self):
        cv2.destroyAllWindows()

    