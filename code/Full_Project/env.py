# Bailey Oteri 
# 05/04/25

import openmc
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import openmc.mgxs as mgxs 
import sys

sys.path.append("data/")
from openmcTemplates import  mgxsBuilder, setTallies

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
        # Delay loading openmc statepoints
        self.ce_sp = None
        self.initial_mg_sp = None
        self.ce_keff = None
        self.initial_mg_keff = None

        self.initial_group_struct = np.logspace(-8,7,num=self.initial_GS_size)
        self.indices = np.linspace(0, self.initial_GS_size - 1, self.final_GS_size, dtype=int)      
        self.collapsedGS_equal_lethargy = self.initial_group_struct[self.indices]
        # Initialize the agent to start at equal lethargy group structure
        self.agent_GS = self.indices
        self.output_path = os.path.join(os.getcwd(), "/data/")
        self.geometry = openmc.Geometry.from_xml(self.output_path + 'geometry.xml')
        self.settings = openmc.Settings.from_xml(self.output_path + 'settings.xml')
        self.batches = self.settings.batches
        self.materials = openmc.Materials.from_xml(self.output_path + 'materials.xml')
        self.tallies_file = setTallies.create_tallies(self.materials, self.initial_group_struct)
        self.episode = 0
        self.step_done = 0
    
    
    
    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)

        if self.ce_sp is None:  # so only loads it once per env
            self.ce_sp = openmc.StatePoint(self.output_path + 'statepoint_ce.h5', autolink=False)
            self.ce_summary =openmc.Summary(self.output_path+'summary_ce.h5')
            self.ce_sp.link_with_summary(self.ce_summary)

            self.ce_keff = self.ce_sp.keff.nominal_value
            self.initial_mg_sp = openmc.StatePoint(self.output_path + 'statepoint_mg.h5', autolink=False)
            self.initial_mg_keff = self.initial_mg_sp.keff.nominal_value
        self.episode += 1
        self.step_done = 0
        self.delta_keff = 1
        # Initialize the agent at the right of the grid
        self.agent_GS = self.indices
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array([self.agent_GS]).astype(np.float32), {}  # empty info dict
    
    def step(self, action):
        self.step_done += 1
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
        agent_keff = self.runOpenMC()
        self.delta_keff = self.ce_keff - agent_keff.nominal_value

        self.delta_keff = round(self.delta_keff, 4)

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
        sp = self.ce_sp
        mgxs_lib = mgxsBuilder.create_mgxs_lib(self.initial_group_struct, self.geometry, self.tallies_file)

        mgxs_lib.load_from_statepoint(sp)
        # Find the names of all of the materials in our run
        all_materials_names = [mat.name for mat in self.geometry.get_all_materials().values()]

        # Initialize MGXS Library with OpenMC statepoint data
        mgxs_file = mgxs_lib.create_mg_library(xs_type='macro', xsdata_names=all_materials_names)
        # Save 200G version of mgxs file for later use
        mgxs_file.export_to_hdf5(self.output_path+'/mgxs_ce.h5')
        # Create a MGXS File which can then be written to disk
        #save mgxs 20 Group version
        collapsedGS = self.initial_group_struct[self.agent_GS]
        groups = mgxs.EnergyGroups(group_edges=collapsedGS)

        mgxs_lib = mgxs_lib.get_condensed_library(groups)

        mgxs_file = mgxs_lib.create_mg_library(xs_type='macro', xsdata_names=all_materials_names)

        # Write the file to disk using the default filename of "mgxs.h5" and move to output folder
        mgxs_file.export_to_hdf5(self.output_path+'/mgxs.h5')
        # Re-define our materials to use the multi-group macroscopic data instead of the continuous-energy data.
        materials_mg, tallies_fileMG = mgxsBuilder.mgxs_materials_build(self.materials, self.output_path, self.agent_GS)

        # Set verbosity to 0 to not have terminal full of OpenMC run data 
        self.settings.verbosity =0
        self.settings.export_to_xml(self.output_path)
        materials_mg.export_to_xml(self.output_path)
        tallies_fileMG.export_to_xml(self.output_path)

        openmc.run(cwd=self.output_path, threads = 10)

        # Read in results of run and return them 
        mg_spfile = os.path.join(self.output_path, 'statepoint_mg.h5')
        os.rename(self.output_path + '/statepoint.' + str(self.batches) + '.h5', mg_spfile)

        mg_sp = openmc.StatePoint(mg_spfile, autolink=False)
        # Get keff for collapsed group run
        agent_keff = mg_sp.keff

        # close statepoint created by run 
        mg_sp._f.close()

        return agent_keff

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        pass
