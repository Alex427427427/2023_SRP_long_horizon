import gymnasium as gym
from maze_gym import MazeGym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from stable_baselines3.common.logger import configure
import os

## create the log folder
def get_next_folder(base_name):
    index = 1
    while True:
        folder = f"{base_name}_{index}"
        if not os.path.exists(folder):
            return folder
        index += 1
# create a folder called log
folder = get_next_folder("log")
os.mkdir(folder)



# create the environment
env = MazeGym(sparse=False, mode="human", move_penalty=0.0, collision_penalty=0.0)
# create the logger
new_logger = configure(folder, ["stdout", "csv", "tensorboard"])

# create the model
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[800, 800], vf=[800, 800])])
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
# set the logger to the model
model.set_logger(new_logger)
# train the model
model.learn(total_timesteps=500000)
# save the model
model.save("models/ppo_maze")

# delete the model
del model # remove to demonstrate saving and loading

# load the model
model = PPO.load("models/ppo_maze")

# roll out once
obs, _ = env.reset()
done = False
truncated = False
while (not done) and (not truncated):
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    env.render(mode="human")