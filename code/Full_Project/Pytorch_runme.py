"""
Bailey Oteri 
04/24/25

REFERENCES: 
    1.) Pytorch Reinforment Learning DQN tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import DQN_Env
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import openmc
import openmc.mgxs as mgxs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import uncertainties
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

# Get time of run start to see how long run takes
start_time = datetime.now().time()

data_path = os.path/join(os.getcwd(), "/data/")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

if device.type == 'cuda':
    print("✅ CUDA is available and will be used.")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
elif device.type == 'mps':
    print("⚠️ Using Apple Metal Performance Shaders (MPS) - this is macOS GPU support.")
else:
    print("❌ CUDA/MPS not available. Using CPU.")

# ===================================================================================================================================
# Read in information about CE run and 200 group runs
# ===================================================================================================================================
# Read in information about CE run and 200 group runs
ce_sp = openmc.StatePoint(data_path +'statepoint_ce.h5', autolink=False)

# Get keff for CE and 200 group structure
ce_keff = ce_sp.keff

# Get RR for CE and 200 group structure
Initial_group_struct = np.logspace(-8,7,num=2001)
indices = np.linspace(0, len(Initial_group_struct) - 1, 21, dtype=int)
collapsedGS = Initial_group_struct[indices]

tally = ce_sp.get_tally(name="absorption_tally")
tally_array = tally.mean
uncertainty = tally.std_dev
    
num_materials = tally.filters[1].num_bins
num_energy_bins = tally.filters[0].num_bins
    
tally_per_bin = tally_array.reshape((num_energy_bins, num_materials))
uncertainty_per_bin = uncertainty.reshape((num_energy_bins, num_materials))

tally200G = np.sqrt((tally_per_bin**2).sum(axis=1))
uncertainty2000G =uncertainty_per_bin.sum(axis=1)

deltaE = np.diff(Initial_group_struct)
ce_rr =tally200G/deltaE
uncertainty2000G /= deltaE

# ===================================================================================================================================
# Create Enviorment
# ===================================================================================================================================
env = Env.GroupCollapseEnv(ce_keff, ce_rr, Initial_group_struct)
# ===================================================================================================================================
# Replay Memory
# ===================================================================================================================================
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
# ===================================================================================================================================
# Q-network
# ===================================================================================================================================
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
# ===================================================================================================================================
# Training
# ===================================================================================================================================

BATCH_SIZE = 8
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 500
TAU = 0.005
LR = 1e-3

# Get number of actions from action space - the amount that each index in collapsedGS can move by in InitialGS
n_actions = env.num_mags*env.num_dirs
# Get the number of state observations
state = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

print("Pre-filling memory with random actions...")
while len(memory) < BATCH_SIZE:
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    action = torch.tensor([[random.randrange(env.action_space)]], device=device, dtype=torch.long)
    obs, reward, done, _ = env.step(action.item(), 0, 0)
    next_state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    reward = torch.tensor([reward.nominal_value if isinstance(reward, uncertainties.UFloat) else reward],
                          dtype=torch.float32).to(device)
    memory.push(state, action, next_state, reward)

steps_done = 0

# Function to select our next action 
def select_action(state):
    global steps_done
    # Epsilon-Greedy Decision: determines how likely the agent will explore (choosing a random action) or exploit (choose best known action)
    # Sample: random number between 0 and 1 to decide whether agent will explore or exploit
    sample = random.random()
    # eps_threshold decays from EPS_START to EPS_END over time
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    # If exploration (random action) is chosen by agent
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    # If exploit (best known action) is chosen by agent, uses DQN to choose best action (highest predicted Q-value)
    else:   
        return torch.tensor([[random.randrange(env.action_space)]], device=device, dtype=torch.long)
 
episode_durations = []
    
# ===================================================================================================================================
# Training Loop
# ===================================================================================================================================
# For saving data of each itteration for post processing
run_history = []
writer = SummaryWriter(log_dir="runs/DQN_" + datetime.now().strftime("%Y%m%d-%H%M%S"))

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return None

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Combine batches and send to GPU once
    non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.tensor(batch.action, device=device).unsqueeze(1)
    reward_batch = torch.cat(batch.reward).to(device)

    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if non_final_next_states.size(0) > 0:
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values.squeeze(), expected_state_action_values)

    # to calc avg loss per episode 
    loss_item = loss.item()
    writer.add_scalar("Loss/train", loss_item, steps_done)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    writer.add_scalar("Loss/train", loss.item(), steps_done)

    # Clip gradients to prevent instability
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)

    optimizer.step()
    return loss_item


# Set number of episodes for max number of itterations want to run
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 50
else:
    num_episodes = 10

global_best_reward = -float('inf')   # Tracks best across all episodes
global_best_structure = None
best_per_episode = []
episode_losses_log = []
# ANSI escape codes for black text and reset
BLACK_TEXT = '\033[30m'
RESET = '\033[0m'

for i_episode in tqdm(range(num_episodes),
                      desc=f"{BLACK_TEXT}Training{RESET}",  # Make desc black
                      unit="episode",
                      bar_format="{l_bar}{bar}{r_bar}",
                      ncols=100,
                      colour="#ff69b4"  # Pink using hex
                     ):
    # Initialize the environment and get its state
    obs = env.reset()
    # Allocate state once per episode
    state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    episode_losses = []
    total_reward = 0
    episode_best_reward = -float('inf')
    episode_best_structure = None
    step_of_best = None

    for t in count():
        action = select_action(state)
        observation, reward, terminated, _ = env.step(action.item(), i_episode, t)

        delta_keff = env.delta_keff
        delta_rr = env.delta_rr

        if isinstance(reward, uncertainties.UFloat):
            reward = reward.nominal_value

        group_structure = env.new_group_structure  
        keff_ngroup = env.mg_keff  
        rr_ngroup = env.mg_rr  

        # Convert reward once and keep on GPU
        reward_tensor = torch.tensor([reward], dtype=torch.float32).to(device)
        writer.add_scalar("Reward/Step", reward, i_episode * 1000 + t)
        writer.add_scalar("Reward/Episode", total_reward, i_episode)
        writer.add_scalar("Delta_keff", delta_keff.nominal_value, i_episode * 1000 + t)
        writer.add_scalar("Delta_RR", delta_rr, i_episode * 1000 + t)

        # Save data to run history for post processing
        run_history.append({
            'Episode': i_episode,
            'Step': t,
            'Action': int(action.item()),
            'Group Structure': group_structure.copy(),
            'keff': keff_ngroup,
            'Reward': float(reward_tensor.item()),
            'Delta_keff': delta_keff.nominal_value,
            'Delta_rr': delta_rr
        })
        df_history = pd.DataFrame(run_history)
        df_history.to_csv("run_history.csv")  

        if float(reward_tensor.item()) > episode_best_reward:
            episode_best_reward = float(reward_tensor.item())
            episode_best_structure = observation.copy()
            step_of_best = t

        done = terminated

        # Only allocate new tensor if episode is not done
        if not done:
            next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            next_state = None

        memory.push(state, action, next_state, reward_tensor)
        state = next_state

        loss_value = optimize_model()
        if loss_value is not None:
            episode_losses.append(loss_value)
      

        # Soft update of the target network’s weights
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            break

    # Save best per episode
    best_per_episode.append({
        'Episode': i_episode,
        'Step of Best in Episode': step_of_best,
        'Total Steps in Episode': t,
        'Best Reward in Episode': episode_best_reward,
        'Best Group Structure in Episode': episode_best_structure
    })
    # Update global best
    if episode_best_reward > global_best_reward:
        global_best_reward = episode_best_reward
        global_best_structure = episode_best_structure

    if episode_losses:
        avg_loss = sum(episode_losses) / len(episode_losses)
        episode_losses_log.appen({
            'Episode': i_episode, 
            "Loss": avg_loss
        })
        writer.add_scalar("Loss/avg_per_episode", avg_loss, i_episode)

writer.close()
# Save data of average loss function per episode
df_losses = pd.DataFrame(episode_losses)
df_losses.to_csv("loss_per_episode.csv", index=False)

# Save data of all episodes as csv file 
df_history = pd.DataFrame(run_history)
df_history.to_csv("final_training_log.csv", index=False)

# Save best output of each episode
df_best_per_episode = pd.DataFrame(best_per_episode)
df_best_per_episode.to_csv("best_per_episode.csv", index=False)

# Save overall best output of all episodes
df_overall_best = pd.DataFrame([{
    "Best Reward": global_best_reward,
    "Best Group Structure": global_best_structure.tolist()
}])
df_overall_best.to_csv("overall_best.csv", index=False)

finish_time = datetime.now().time()
print('Start Time was:', start_time)
print('Finish Time was:', finish_time)
