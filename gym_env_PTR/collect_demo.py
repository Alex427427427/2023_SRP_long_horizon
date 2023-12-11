import gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from huggingface_sb3 import load_from_hub

from imitation.data.types import save
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
import argparse

parser = argparse.ArgumentParser(description='Collect demonstrations for imitation learning')
parser.add_argument('--logdir',type=str,default='./logs/',help="Folder for logging")
parser.add_argument('--env',type=str,default='Pendulum-v1',help="gym environment")
parser.add_argument('--steps',type=int,default=500000,help="Number of training steps")
parser.add_argument('--demos',type=int,default=100,help="Number of demos")

args = parser.parse_args()

if __name__ == "__main__":

    env = gym.make(args.env) # make the environment
    rng = np.random.default_rng(0)
    env.seed(0)

    if args.env == 'MountainCarContinuous-v0':
        # create expert policy
         expert = PPO(
            policy=MlpPolicy,
            env=env,
            use_sde=True, # use sde for mountaincar
            seed=42,
            tensorboard_log=args.logdir+'/expert_'+args.env+'_%d'%args.steps
        )
    else:
         expert = PPO(
            policy=MlpPolicy,
            env=env,
            seed=42,
            tensorboard_log=args.logdir+'/expert_'+args.env+'_%d'%args.steps
        )
       

    mean_reward, std_reward = evaluate_policy(expert, env, n_eval_episodes=10,render=False) # stable baseline 3 function
    # returns mean and std of reward over 10 episodes
    print(f"Before training: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    expert.learn(args.steps) # train the expert policy

    # collect samples
    print("Sampling expert transitions.")
    # in the format of imitation.data.types.Trajectories, later stored in a .npz file
    rollouts = rollout.rollout(
        expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        rollout.make_sample_until(min_timesteps=None, min_episodes=args.demos),
        rng=rng,
    )

    # evaluate policy again, store the mean and std of reward in a .npy file
    mean_reward, std_reward = evaluate_policy(expert, env, n_eval_episodes=100,render=False)

    print(f"After training: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    save(args.logdir+'demos_'+args.env+'%d_%d.npz'%(args.steps,args.demos),rollouts)
    np.save(args.logdir+'expert_perf_'+args.env+'%d_%d.npy'%(args.steps,args.demos),np.array([mean_reward,std_reward]))