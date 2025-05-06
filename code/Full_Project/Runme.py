# Bailey Oteri 
# 05/05/25
# CSCI 575 FINAL PROJECT
# MGO- PPO
# Runs with OpenMC and does full model creation

from stable_baselines3.common.env_checker import check_env
from env import MGXS_Collapse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import time
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
path = "ENTER_PATH_TO_SAVE_MODEL"

# Define function to create environment instances
def make_env(seed):
    def _init():
        env = MGXS_Collapse()
        env.seed(seed)
        return env
    return _init

if __name__ == "__main__":
    # Validate the environment
    #env_test = MGXS_Collapse()
    #check_env(env_test, warn=True)

    # Evaluation env
    eval_env = MGXS_Collapse()

    n_procs = 4
    NUM_EXPERIMENTS = 1
    TRAIN_STEPS = 100
    EVAL_EPS =  50

    reward_averages = []
    reward_std = []
    training_times = []

    #vec_env = SubprocVecEnv([make_env(i) for i in range(n_procs)], start_method="spawn")
    vec_env = DummyVecEnv([make_env(i) for i in range(n_procs)])

    rewards = []
    times = []

    for experiment in range(NUM_EXPERIMENTS):
        vec_env.reset()
        model = PPO("MlpPolicy", vec_env, verbose=1, device='cpu')
        start = time.time()
        model.learn(total_timesteps=TRAIN_STEPS)
        times.append(time.time() - start)

        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
        rewards.append(mean_reward)

    model.save(path)

    vec_env.close()
    reward_averages.append(np.mean(rewards))
    reward_std.append(np.std(rewards))
    training_times.append(np.mean(times))

    print(f"Average Reward: {reward_averages[-1]:.2f}")
    print(f"Reward Std Dev: {reward_std[-1]:.2f}")
    print(f"Average Training Time: {training_times[-1]:.2f} sec")
