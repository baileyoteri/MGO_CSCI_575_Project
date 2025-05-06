# Bailey Oteri 
# 05/04/25
# Create custom env following gym structure for SB3 for MGXS collapse project with PPO multiprocessing

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MGXS_Collapse(gym.Env): 
    # Custom enviorment following gym interface
     
    metadata = {"render_modes": ["human"]}

    # Define directions for cleaner code 
    LEFT = 0
    RIGHT = 1

    def __init__(self, initial_GS_size=2000, render_mode="human"):
        super(MGXS_Collapse, self).__init__()
        self.render_mode = render_mode

        # Size of group structures
        self.initial_GS_size = initial_GS_size+1
        self.final_GS_size = 20+1

        # Define action and observation space
        # They must be gym.spaces objects
        # 2 actions, plus one or minus one from chosen index
        self.n_boundries = self.final_GS_size - 2
        self.n_directions = 2
        self.action_space = spaces.Discrete(self.n_boundries*self.n_directions)
        self.delta_keff = 1
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(
            low=0, high=self.initial_GS_size, shape=(1,21), dtype=np.float32
        )

        self.initial_group_struct = np.logspace(-8,7,num=self.initial_GS_size)
        self.indices = np.linspace(0, self.initial_GS_size - 1, self.final_GS_size, dtype=int)      
        self.collapsedGS_equal_lethargy = self.initial_group_struct[self.indices]
        # Initialize the agent to start at equal lethargy group structure
        self.agent_GS = self.indices

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
 
        self.delta_keff = 1
        # Initialize the agent at the right of the grid
        self.agent_GS = self.indices
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array([self.agent_GS]).astype(np.float32), {}  # empty info dict
    
    def step(self, action):
        direction = action // self.n_boundries
        boundary = action % self.n_boundries
        boundary +=1 # skip first and last boundaries 

        if direction == self.LEFT:
            self.agent_GS[boundary] -= 1
        elif direction == self.RIGHT:
            self.agent_GS[boundary] += 1
        else:
            raise ValueError(
                f"Received invalid action={action} which is not part of the action space"
            )
        
        # Check that boundries are all valid still 
        if np.all(np.diff(self.agent_GS) > 0) and np.all((self.agent_GS >= 0) & (self.agent_GS <= self.initial_GS_size - 1)):
            # All conditions passed, run OpenMC
            pass 
        else:
            # One or more conditions failed, undo action and do same action in opposite direction 
            if action == self.LEFT: 
                self.agent_GS[boundary] += 2
            elif action == self.RIGHT: 
                self.agent_GS[boundary] -= 2
            pass


        truncated = False  # we do not limit the number of steps here

        # Run OpenMG with new group structure once it is verified to be valied and get new keff value
        self.delta_keff = self.runOpenMC()
        self.delta_keff = round(self.delta_keff, 4)
        
        #self.delta_keff = self.ce_keff - agent_keff

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = 1 if self.delta_keff == 0 else 0

        # Optionally we can pass additional info, we are not using that for now
        info = {}
        # End the run if goal is found
        terminated = bool(self.delta_keff == 0)

        return (
            np.array([self.agent_GS]).astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )
    
    def runOpenMC(self):
        fake_keff = self.delta_keff - 0.01
        return fake_keff
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        pass
