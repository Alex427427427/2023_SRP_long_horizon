from maze_gym import MazeGym
import torch

from stable_baselines3 import PPO


env = MazeGym()
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[800, 800], vf=[800, 800])])
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model = PPO.load("models/ppo_maze")
print(model.policy)

# plot the 

episodes = 500

for episode in range(episodes):
	terminated = False
	obs, _ = env.reset()
	while not terminated:
		action, _states = model.predict(obs)
		env.render()
		#print("action", action)
		obs, reward, terminated, truncated, info = env.step(action)
		#print('reward',reward)
		
env.close()