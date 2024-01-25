from stable_baselines3.common.env_checker import check_env
from maze_gym import MazeGym

env = MazeGym()
check_env(env)