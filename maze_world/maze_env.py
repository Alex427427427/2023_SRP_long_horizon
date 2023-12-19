# create a class for the environment

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import os
import glob

# create a function to take in png and convert into a numpy array
def png_to_numpy(png_file):
    # read in the png file
    img = plt.imread(png_file)
    # convert to grayscale
    img = np.mean(img,axis=2)
    # convert to binary
    img = (img < 0.5).astype(np.float32)
    return img

if __name__ == "__main__":
    maze_filename = "mazes/maze_1.png"
    # convert into numpy array, display the maze
    maze = png_to_numpy(maze_filename)
    plt.imshow(maze.T,origin='lower',cmap='gray')
    plt.show()

class Maze():
    # note the array is flipped when plotted
    def __init__(self,sparse=True,model=None, move_penalty=0.05, goal_reward=10.0, collision_penalty=0.5):
        self.N = 40 # sidelength of the grid map
        
        x = np.arange(self.N) # create x coord array
        y = np.arange(self.N) # create y coord array

        self.xx,self.yy = np.meshgrid(x,y) # create matrices of x and y coords, as separate matrices, that can be served as input to 
        # some multidimensional function

        # create an occupancy map
        self.occ_map = np.zeros((self.N,self.N))
        self.occ_map = png_to_numpy("mazes/maze_1.png")
        

        # goal location
        self.xl = 11 
        self.yl = 18
        
        # reward
        self.sparse = sparse
        #self.fv = lambda x,y : np.exp(-1*((x-self.xl)**2+(y-self.yl)**2)) # e^(-dist^2). gaussian proximity reward function
        if sparse or model is None:
            #self.f = self.fv(self.yy, self.xx) # reward vector
            self.f = np.zeros((self.N, self.N))
            self.f[self.xl, self.yl] = goal_reward
        else:
            self.model = model
            # create a tensor of the grid
            grid = torch.tensor(np.stack([self.yy.flatten(), self.xx.flatten()], axis=1), dtype=torch.float)
            # get the predictions
            preds = model.predict_time_to_goal(grid).detach().numpy()
            #print(preds)
            # reshape the predictions
            preds = preds.reshape(self.N, self.N)
            #rewards = np.exp(-1*preds)
            preds = (preds - np.min(preds))/(np.max(preds) - np.min(preds))
            self.times = preds
            rewards = 1 - preds
            self.f = rewards

        # random initial location outside the box
        # continually attempt to find a random initial location outside the box until one is found
        self.state = self.search_for_free_state()
        
        self.action_space = np.array([[0,1],[0,-1],[-1,0],[1,0]]) # up down left right
        self.move_penalty = move_penalty
        self.collision_penalty = collision_penalty
        self.goal_reward = goal_reward
        
    def search_for_free_state(self):
        while True:
            x0 = np.random.randint(self.N)
            y0 = np.random.randint(self.N)
            # if the location is not an obstacle, is not the goal, and is not in the box, then break
            obs = self.occ_map[x0,y0]
            goal = (x0 == self.xl) and (y0 == self.yl)
            #box = (x0 >= 8) and (x0 <= 10) and (y0 >= 6) and (y0 <= 16)
            if (obs == 0) and (not goal):
                break
        return np.array([x0,y0])
    # apply action and update the state
    def mm(self,X,u):
        collision = False
        x = X[0] + u[0]
        y = X[1] + u[1]
        # if the new state is at an obstacle or out of the map, stay at the same state
        if x < 0 or x > self.N-1 or y < 0 or y > self.N-1 or self.occ_map[x,y] == 1:
            collision = True
            return X, collision
        else:
            return np.array([x,y]), collision
    
    # step one interaction
    def step(self,idx):
        done = False
        u = self.action_space[idx,:]
        new_state, collision = self.mm(self.state,u)
        self.state = np.copy(new_state)
        # find out if the agent has reached the goal
        if self.state[0] == self.xl and self.state[1] == self.yl:
            done = True
        # compute reward
        reward = self.f[new_state[0],new_state[1]]
        # subtract movement penalty
        reward = reward - self.move_penalty
        # subtract collision penalty
        if collision:
            reward = reward - self.collision_penalty
        # return new state, gaussian proximity reward
        return self.state, reward, done
    
    def reset(self):
        self.state = self.search_for_free_state()
        return np.copy(self.state)
    
    def plot(self):
        state = np.copy(self.state)+0.5
        plt.figure(figsize=(5,5))
        plt.imshow(self.occ_map.T, origin="lower", cmap='gray')
        plt.plot(state[0],state[1],'ro') # agent location
        plt.plot(self.xl,self.yl,'gx') # goal location
        plt.axis('off')
        plt.show()

    def plot_reward(self):
        state = np.copy(self.state)+0.5
        plt.figure(figsize=(5,5))
        plt.imshow(self.f.T, origin="lower", cmap='gray')
        plt.plot(self.xl,self.yl,'gx') # goal location
        plt.show()
        #plt.plot(state[0],state[1],'ro') # agent location

    def plot_reward_and_trajectory(self, trajectory):
        state = np.copy(self.state)+0.5
        plt.figure(figsize=(5,5))
        plt.imshow(self.f.T, origin="lower", cmap='gray')
        plt.plot(state[0],state[1],'ro') # agent location
        plt.plot(trajectory[:, 0]+0.5, trajectory[:, 1]+0.5, 'b-')

    def shrink_reward(self, shrink_factor):
        self.f *= np.exp(-shrink_factor*(self.times)**2)

    # takes in the env, Q table, episode length
    # plays one episode
    # plots the reward and q value for every step. 
    def test_value(self,Q,steps=50,disp=True, greediness=1.0):
        state = self.reset()
        xm = [] # state
        rsum = 0 # total reward
        if self.sparse:
            reward_title = "True reward"
        else:
            reward_title = "Expanded reward"

        for j in range(steps):
            if np.random.rand() < greediness:
                a = np.argmax(Q[self.N*state[0]+state[1],:])
            else:
                a = np.random.randint(self.action_space.shape[0])

            state, reward, done = self.step(a)
        
            xm.append(np.copy(state))
            rsum = rsum+reward
            if disp:
                s = np.copy(self.state)+0.5
                plt.subplot(1,2,1)
                plt.imshow(self.f.T, origin="lower",extent=[0,self.N,0,self.N], cmap='gray')
                plt.plot(self.xl+0.5,self.yl+0.5,'gx') # goal location
                plt.title(reward_title)
                plt.plot(np.vstack(xm)[:,0]+0.5,np.vstack(xm)[:,1]+0.5,'c-o')
                plt.plot(s[0],s[1],'ro') # agent location
                plt.axis('off')

                plt.subplot(1,2,2)
                plt.imshow(np.max(Q,axis=1).reshape(self.N,self.N).T, origin="lower",extent=[0,self.N,0,self.N], cmap='gray')
                plt.plot(self.xl+0.5,self.yl+0.5,'gx') # goal location
                plt.title('Q estimate')
                plt.plot(np.vstack(xm)[:,0]+0.5,np.vstack(xm)[:,1]+0.5,'c-o')
                plt.plot(s[0],s[1],'ro') # agent location
                plt.axis('off')
                plt.title("Q value")

                display.clear_output(wait=True)
                plt.show()
            if done:
                break
        return rsum, xm
    
    def value_iteration(self,init=None,num_eps=1000,ep_length=1000,alpha=0.9,gamma=0.9,final_greediness=0.5,eps_anneal=True,plot_freq=1000,disp=True, reward_shrink=0.0,
                        save_folder=None):
        if save_folder is not None:
            for f in glob.glob(f"{save_folder}/*"):
                os.remove(f)
        if init is None:
            Q = np.ones((self.N*self.N,self.action_space.shape[0])) # initialise Q table
        else:
            Q = init
        reward = 0
        if disp:
            plt.figure(figsize=(15,5))
        xm = []
        rewards = []
        k = 0
        for i in range(num_eps):
            # reset the environment
            self.reset()
            
            # take time steps.
            for j in range(ep_length):
                # observe.
                state = np.copy(self.state)
                
                # calculate greediness.
                if eps_anneal:
                    greediness = final_greediness*np.exp(1-ep_length/(j+1)) # asymptotically rising exponential to the final greediness
                else:
                    greediness = final_greediness
                
                # choose action.
                if np.random.rand() < greediness:
                    a = np.argmax(Q[self.N*state[0]+state[1],:])
                else:
                    a = np.random.randint(self.action_space.shape[0])

                # take action. receive reward. observe new state.
                new_state, new_reward, done = self.step(a)
                xm.append(np.copy(new_state))

                # choose best action at the new state, according to current knowledge. 
                new_a = np.argmax(Q[self.N*new_state[0]+new_state[1],:])
                # get the Q value of the best action chosen at the new state. 
                Qmax = Q[self.N*new_state[0]+new_state[1],new_a]
                # update the Q value of the current state and action. alpha is step size. use the reward of the previous step. 
                Q[self.N*state[0]+state[1],a] = (1-alpha)*Q[self.N*state[0]+state[1],a] + alpha*(new_reward + gamma*Qmax)
                reward = new_reward
                
                rewards.append(reward)

                # shrink the reward
                if reward_shrink > 0:
                    if j % plot_freq == 0:
                        self.f *= np.exp(-reward_shrink*(self.times)**2)
                
                if self.sparse:
                    reward_title = "True reward"
                else:
                    reward_title = "Expanded reward"
                
                
                if (j %plot_freq == 0) and (disp):
                    s = np.copy(self.state)
                    plt.subplot(2,3,1)
                    plt.imshow(self.f.T,origin='lower', cmap='gray')
                    plt.plot(s[0],s[1],'ro') # agent location
                    plt.axis('off')
                    plt.colorbar()
        #             plt.plot(np.vstack(xm)[:,0],np.vstack(xm)[:,1])
                    plt.title(reward_title, fontsize=7)

                    plt.subplot(2,3,2)
                    plt.imshow(np.max(Q,axis=1).reshape(self.N,self.N).T,origin='lower', cmap='gray')
                    plt.plot(s[0],s[1],'ro') # agent location
                    plt.axis('off')
                    plt.colorbar()
        #             plt.plot(np.vstack(xm)[:,0],np.vstack(xm)[:,1])
                    plt.title('Q value')

                    plt.subplot(2,3,3)
                    plt.imshow(np.argmax(Q,axis=1).reshape(self.N,self.N).T,origin='lower', cmap='gray')
                    plt.plot(s[0],s[1],'ro') # agent location
                    plt.axis('off')
        #             plt.plot(np.vstack(xm)[:,0],np.vstack(xm)[:,1])
                    plt.title('Best action')

                    plt.subplot(2,1,2)
                    plt.plot(rewards,'o',alpha=0.01)
                    plt.ylabel('Reward')
                    plt.xlabel('Env interaction')
                    if save_folder is not None:
                        plt.savefig(f"{save_folder}/img_{k}.png")
                    display.clear_output(wait=True)
                    plt.show()
                    k += 1
                    print (f"Greediness: {greediness}")
        return Q
    
    def value_iter2(self,init=None,iters=10000,alpha=0.9,gamma=0.9,initial_eps=1.0,eps_anneal_rate=0.0,plot_freq=1000,disp=True, reward_shrink=0.0,
                        save_folder=None):
        if save_folder is not None:
            for f in glob.glob(f"{save_folder}/*"):
                os.remove(f)
        if init is None:
            Q = np.ones((self.N*self.N,self.action_space.shape[0])) # initialise Q table
        else:
            Q = init
        reward = 0
        if disp:
            plt.figure(figsize=(15,5))
        xm = []
        rewards = []
        k = 0
        for j in range(iters):
            state = np.copy(self.state)
            
            # Epsilon-greedy
            if eps_anneal_rate > 0:
                greediness = 1 - initial_eps*np.exp(-j/iters*eps_anneal_rate) # asymptotically rising exponential to the final greediness
            else:
                greediness = 1 - initial_eps*np.exp(-j/iters*eps_anneal_rate)
                
            if np.random.rand() < greediness:
                a = np.argmax(Q[self.N*state[0]+state[1],:])
            else:
                a = np.random.randint(self.action_space.shape[0])

            new_state, new_reward, done = self.step(a)
            xm.append(np.copy(new_state))

            new_a = np.argmax(Q[self.N*new_state[0]+new_state[1],:])
            Qmax = Q[self.N*new_state[0]+new_state[1],new_a]
            Q[self.N*state[0]+state[1],a] = (1-alpha)*Q[self.N*state[0]+state[1],a] + alpha*(new_reward + gamma*Qmax)
            reward = new_reward
            
            rewards.append(reward)

            # shrink the reward
            if reward_shrink > 0:
                if j % plot_freq == 0:
                    self.f *= np.exp(-reward_shrink*(self.times)**2)
            
            if self.sparse:
                reward_title = "True reward"
            else:
                reward_title = "Expanded reward"
            
            
            if (j %plot_freq == 0) and (disp):
                s = np.copy(self.state)
                plt.subplot(2,3,1)
                plt.imshow(self.f.T,origin='lower', cmap='gray')
                plt.plot(s[0],s[1],'ro') # agent location
                plt.axis('off')
                plt.colorbar()
    #             plt.plot(np.vstack(xm)[:,0],np.vstack(xm)[:,1])
                plt.title(reward_title, fontsize=7)

                plt.subplot(2,3,2)
                plt.imshow(np.max(Q,axis=1).reshape(self.N,self.N).T,origin='lower', cmap='gray')
                plt.plot(s[0],s[1],'ro') # agent location
                plt.axis('off')
                plt.colorbar()
    #             plt.plot(np.vstack(xm)[:,0],np.vstack(xm)[:,1])
                plt.title('Q value')

                plt.subplot(2,3,3)
                plt.imshow(np.argmax(Q,axis=1).reshape(self.N,self.N).T,origin='lower', cmap='gray')
                plt.plot(s[0],s[1],'ro') # agent location
                plt.axis('off')
    #             plt.plot(np.vstack(xm)[:,0],np.vstack(xm)[:,1])
                plt.title('Best action')

                #plt.subplot(2,1,2)
                #plt.plot(rewards,'o',alpha=0.01)
                #plt.ylabel('Reward')
                #plt.xlabel('Env interaction')
                if save_folder is not None:
                    plt.savefig(f"{save_folder}/img_{k}.png")
                display.clear_output(wait=True)
                plt.show()
                k += 1
                print (f"Greediness: {greediness}")
        return Q
    
    def evaluate_Q(self, Q, episodes=500, steps=100, initial_eps=1.0, eps_anneal_rate=0.7):
        goals_reached = 0
        for i in range(episodes):
            self.reset()
            for j in range(steps):
                # Epsilon-greedy
                if eps_anneal_rate > 0:
                    greediness = 1 - initial_eps*np.exp(-j/steps*eps_anneal_rate)
                else:
                    greediness = 1 - initial_eps*np.exp(-j/steps*eps_anneal_rate)

                if np.random.rand() < greediness:
                    a = np.argmax(Q[self.N*self.state[0]+self.state[1],:])
                else:
                    a = np.random.randint(self.action_space.shape[0])

                new_state, new_reward, done = self.step(a)
                if done:
                    goals_reached += 1
                    break
            
        return goals_reached/episodes