import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

def eval(model):
    eval_env = gym.make("CartPole-v1", render_mode="rgb_array")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)

    return mean_reward, std_reward

def main():
    train_env = gym.make("CartPole-v1", render_mode="rgb_array")

    model = PPO(MlpPolicy, train_env, verbose=0)
    model.learn(total_timesteps=50_000)

    mean_reward, std_reward = eval(model)

    print(f"mean reward: {mean_reward:.3f} +/- {std_reward:.3f}")

    


if __name__ == "__main__":
    main()
