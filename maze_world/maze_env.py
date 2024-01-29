# create a class for the environment

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import os
import glob


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

# given a coordinate, returns whether the coordinate is in the start zone, indicated by green pixels. 
def is_start_zone(image_path, x, y):
    # green pixel is the start zone
    img = plt.imread(image_path)
    height, width, channels = img.shape
    rgb = img[x, y]
    red = rgb[0]
    green = rgb[1]
    blue = rgb[2]
    if red < 0.3 and green > 0.7 and blue < 0.3:
        return True
    else:
        return False

image_path = "mazes/maze_procedural_1.png"

# a reinforcement learning environment, a 2D maze. 
class Maze():
    # note the array is flipped when plotted
    def __init__(self,sparse=True,model=None, move_penalty=0.05, goal_reward=10.0, collision_penalty=0.5):
        # create an occupancy map
        self.occ_map = png_to_occ_map(image_path)
        self.collision_penalty = collision_penalty

        self.Nx = self.occ_map.shape[0]
        self.Ny = self.occ_map.shape[1]

        # goal location and reward
        goal_tuple = find_goal_location(image_path)
        self.gx = goal_tuple[0] 
        self.gy = goal_tuple[1]
        self.goal_reward = goal_reward

        # randomly select a free state
        self.state = self.free_state_search()

        # actions: up down left right
        self.action_space = np.array([[0,1],[0,-1],[-1,0],[1,0],[0,0]]) # up down left right stay
        self.move_penalty = move_penalty

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
    
    # returns a random state free from obstacle from the start zone
    def start_zone_search(self):
        while True:
            x0 = np.random.randint(14,26)
            y0 = np.random.randint(12)
            #if the location is not an obstacle, is not the goal, then break
            obs = self.occ_map[x0,y0]
            goal = (x0 == self.gx) and (y0 == self.gy)
            if (obs == 0) and (not goal):
                break
        return np.array([x0,y0])

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
    
    def reset_to_start_zone(self):
        self.state = self.start_zone_search()
        return np.copy(self.state)
        
    def reset_to_free_state(self):
        self.state = self.free_state_search()
        return np.copy(self.state)

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
        # returns new state, reward, done
    def step(self,idx):
        done = False

        # select action
        u = self.action_space[idx,:]
        # calculate next state and whether a collision has occured
        new_state, collision = self.movement_control(self.state,u)
        # overwrite the state
        self.state = np.copy(new_state)

        # find out if the agent has reached the goal
        if self.state[0] == self.gx and self.state[1] == self.gy:
            done = True

        # compute reward
        reward = self.reward_landscape[new_state[0],new_state[1]]
        # subtract movement penalty if the agent has not stayed still
        if idx != 4:
            reward = reward - self.move_penalty
        # subtract collision penalty
        if collision:
            reward = reward - self.collision_penalty

        return self.state, reward, done
    
    # plotting functions
    def plot(self):
        # add 0.5 so that the state is plotted in the middle of the grid square
        state = np.copy(self.state)+0.5

        plt.figure(figsize=(5,5))
        plt.imshow(self.occ_map.T, origin="lower", cmap='gray') # plot the occupancy map
        plt.plot(state[0],state[1],'ro') # agent location
        plt.plot(self.gx,self.gy,'gx') # goal location
        plt.axis('off')
        plt.show()

    def plot_reward(self):
        # add 0.5 so that the state is plotted in the middle of the grid square
        state = np.copy(self.state)+0.5

        plt.figure(figsize=(5,5))
        # plot the reward landscape
        plt.imshow(self.reward_landscape.T, origin="lower", cmap='gray') # plot the reward landscape, along with the occupancy map
        plt.plot(self.gx,self.gy,'gx') # goal location
        plt.show()
    
    # takes in the env, Q table, episode length
    # plays one episode
    # plots the reward and q value for every step. 
    # returns the trajectory and whether the goal is reached
    def test_Q_once(self, Q, episode_length=200, final_greediness=0.5, eps_anneal=True, use_start_zone=False, disp=False):
        if use_start_zone:
            state = self.reset_to_start_zone()
        else:
            state = self.reset_to_free_state()

        # select the appropriate title
        if self.sparse:
            reward_title = "True reward"
        else:
            reward_title = "Expanded reward"

        trajectory = [] # list of states visited in order
        success = False

        for j in range(episode_length):
            # calculate greediness.
            if eps_anneal:
                greediness = final_greediness*np.exp(1-episode_length/(j+1)) # rising exponential to the final greediness
            else:
                greediness = final_greediness

            # e-greedily select action
            if np.random.rand() < greediness:
                a = np.argmax(Q[self.Ny*state[0]+state[1],:])
            else:
                a = np.random.randint(self.action_space.shape[0])

            # interact
            state, reward, done = self.step(a)
            trajectory.append(np.copy(state))

            if disp:
                s = np.copy(self.state)+0.5 # add 0.5 so that the dot shows up at the center of the grid
                plotting_trajectory = np.vstack(trajectory)+0.5 # add 0.5 so that the trajectory shows up at the center of the grid

                plt.subplot(1,2,1)
                plt.imshow(self.reward_landscape.T, origin="lower",extent=[0,self.Nx,0,self.Ny], cmap='gray')
                plt.plot(self.gx+0.5,self.gy+0.5,'gx') # goal location
                plt.title(reward_title)
                plt.plot(plotting_trajectory[:,0],plotting_trajectory[:,1],'c-o') # add 0.5 for plotting
                plt.plot(s[0],s[1],'ro') # agent location
                plt.axis('off')

                plt.subplot(1,2,2)
                plt.imshow(np.max(Q,axis=1).reshape(self.Nx,self.Ny).T, origin="lower",extent=[0,self.Nx,0,self.Ny], cmap='gray')
                plt.plot(self.gx+0.5,self.gy+0.5,'gx') # goal location
                plt.title('Q estimate')
                plt.plot(plotting_trajectory[:,0],plotting_trajectory[:,1],'c-o') # add 0.5 for plotting
                plt.plot(s[0],s[1],'ro') # agent location
                plt.axis('off')
                plt.title("Q value")

                display.clear_output(wait=True)
                plt.show()

            # if the goal is reached, break
            if done:
                success = True
                break
        return trajectory, success
    
    # takes in the env, Q table, episode length
    # plays over multiple episodes with specified length, and annealing epsilon
    # returns a single number indicating the fraction of episodes where the goal is reached
    def evaluate_Q(self, Q, episodes=500, episode_length=100, final_greediness=0.5,eps_anneal=True, use_start_zone=False):
        goals_reached = 0
        for i in range(episodes):
            traj, success = self.test_Q_once(Q, episode_length=episode_length, disp=False, 
                                             final_greediness=final_greediness, eps_anneal=eps_anneal, use_start_zone=use_start_zone)
            if success:
                goals_reached += 1
        return goals_reached/episodes
    
    # train Q for 1 epoch
    def train_Q_1_epoch(self, init_Q=None, episodes=500, episode_length=500, use_start_zone=False,
                        alpha=0.9,gamma=0.9,
                        final_greediness=0.5,eps_anneal=True,
                        reward_shrink=0.0, shrink_freq=10,
                        plot_freq=10, disp=True, save_folder=None):
        
        if disp:
            if save_folder is not None:
                images_saved = 0 # counter for saving images
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                # clear the save folder
                for f in glob.glob(f"{save_folder}/*"):
                    os.remove(f)
            plt.figure(figsize=(15,5))
            # choose the right reward title
            if self.sparse:
                reward_title = "True reward"
            else:
                reward_title = "Expanded reward"

        # initialise Q table
        if init_Q is None:
            Q = np.ones((self.Nx*self.Ny,self.action_space.shape[0])) # _Qialise Q table
        else:
            Q = init_Q
        
        reward = 0
        trajectory = []
        rewards = []
        
        for i in range(episodes):
            # reset the environment
            if use_start_zone:
                self.reset_to_start_zone()
            else:
                self.reset_to_free_state()

            # shrink the reward
            if reward_shrink > 0:
                if i % shrink_freq == 0:
                    self.shrink_reward(reward_shrink)
            
            # take time steps.
            for j in range(episode_length):
                # observe.
                state = np.copy(self.state)
                
                # calculate greediness.
                if eps_anneal:
                    greediness = final_greediness*np.exp(1-episode_length/(j+1)) # asymptotically rising exponential to the final greediness
                else:
                    greediness = final_greediness
                
                # choose action.
                if np.random.rand() < greediness:
                    a = np.argmax(Q[self.Ny*state[0]+state[1],:])
                else:
                    a = np.random.randint(self.action_space.shape[0])

                # take action. receive reward. observe new state.
                new_state, new_reward, done = self.step(a)
                trajectory.append(np.copy(new_state))

                # choose best action at the new state, according to current knowledge. 
                new_a = np.argmax(Q[self.Ny*new_state[0]+new_state[1],:])
                # get the Q value of the best action chosen at the new state. 
                Qmax = Q[self.Ny*new_state[0]+new_state[1],new_a]
                # update the Q value of the current state and action. alpha is step size. use the reward of the previous step. 
                Q[self.Ny*state[0]+state[1],a] = (1-alpha)*Q[self.Ny*state[0]+state[1],a] + alpha*(new_reward + gamma*Qmax)
                reward = new_reward
                rewards.append(reward)
                
            # every few episodes, plot the reward landscape, Q value, and best action
            if (i % plot_freq == 0) and (disp):

                plt.subplot(2,3,1)
                plt.imshow(self.reward_landscape.T,origin='lower', cmap='gray')
                plt.axis('off')
                plt.title(reward_title)

                plt.subplot(2,3,2)
                plt.imshow(np.max(Q,axis=1).reshape(self.Nx,self.Ny).T,origin='lower', cmap='gray')
                plt.axis('off')
                plt.title('Q value')

                plt.subplot(2,3,3)
                plt.imshow(np.argmax(Q,axis=1).reshape(self.Nx,self.Ny).T,origin='lower', cmap='gray')
                plt.axis('off')
                plt.title('Best action')

                plt.subplot(2,1,2)
                plt.plot(rewards,'ko',alpha=0.01)
                plt.ylabel('Reward')
                plt.xlabel('Env interaction')

                if save_folder is not None:
                    plt.savefig(f"{save_folder}/img_{images_saved}.png")
                    images_saved += 1

                display.clear_output(wait=True)
                plt.show()
        return Q
    
    
if __name__ == "__main__":

    maze = Maze()
    maze.plot()