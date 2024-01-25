from maze_gym import MazeGym


env = MazeGym()
episodes = 50

for episode in range(episodes):
	terminated = False
	obs = env.reset()
	while not terminated:
		random_action = env.action_space.sample()
		env.render()
		print("action",random_action)
		obs, reward, terminated, truncated, info = env.step(random_action)
		print('reward',reward)
		
env.close()