import gymnasium as gym
from maze_gym import MazeGym
from maze_PTR_model import MazePTRModel

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

model = MazePTRModel()
model.load_state_dict(torch.load("last_model.pt"))
# Parallel environments
env = MazeGym(sparse=False, mode="human", move_penalty=0.0, collision_penalty=0.0)


policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[800, 800], vf=[800, 800])])
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=500000)
model.save("ppo_maze")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_maze")

obs, _ = env.reset()
done = False
truncated = False
while (not done) and (not truncated):
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    env.render(mode="human")