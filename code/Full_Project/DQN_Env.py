"""
Bailey Oteri 
04/17/25
"""
import numpy as np
import os  # to create folder and environment variable
import sys
import openmc  # for the python API of OpenMC 
import openmc.mgxs as mgxs # for multigroup cross section mode

sys.path.append("data/")
from openmcTemplates import  mgxsBuilder,setTallies

class GroupCollapseEnv:
    def __init__(self, ce_keff, ce_rr, group_boundaries_2k, n_groups=5):
        self.ce_keff = ce_keff
        self.ce_rr = ce_rr
        self.group_boundaries_2k = group_boundaries_2k
        self.n_groups = n_groups
        #self.initial_groups_energies =np.logspace(-8,7,num=201)
        self.min_idx = 0
        self.max_idx = len(group_boundaries_2k)-1
        self.output_path = os.path.join(os.getcwd(), "/data/")

        # Total actions = (n_groups - 1) boundaries × 2 directions × 10 magnitudes
        self.num_boundaries = n_groups - 1
        self.num_dirs = 2
        self.num_mags = 200
        self.action_dim = (self.n_groups - 1) * self.num_dirs * self.num_mags
        self.geometry = openmc.Geometry.from_xml(self.output_path + 'geometry.xml')
        self.settings = openmc.Settings.from_xml(self.output_path + 'settings.xml')
        self.batches = self.settings.batches
        self.materials = openmc.Materials.from_xml(self.output_path + 'materials.xml')
        self.ce_sp = os.path.join(os.getcwd(), 'statepoint_ce.h5')
        self.ce_summary = os.path.join(os.getcwd(), 'summary_ce.h5')
        self.tallies_file = setTallies.create_tallies(self.materials, group_boundaries_2k)
        # create delta for first run of episode
        self.delta_keff = 1 
        self.delta_rr = 1
        self.max_steps = 5
        self.reset()

    # Called at the start of each episode to initialize it
    def reset(self):
        self.state = np.linspace(self.min_idx, self.max_idx, self.n_groups + 1, dtype=int)
        self.steps_taken = 0

        return self.state.copy()
    
    # Function that logs all errors that occur when finding a new group structure to an txt file
    def log_error(message, episode=None, step=None, filename="error_log.err"):
        with open(filename, "a") as f:
            f.write("\n" + "="*40 + "\n")
            if episode is not None and step is not None:
                f.write(f"📘 Episode: {episode}, Step: {step}\n")
            f.write("❌ Error: " + message + "\n")
            f.write("="*40 + "\n")

    def step(self, action, episode, step):
        idx, direction, magnitude = self.decode_action(action)
        new_state = self.state.copy()

        shift = direction * magnitude

        try:
            new_state[idx] = np.clip(
                new_state[idx] + shift,
                new_state[idx - 1] + 1,
                new_state[idx + 1] - 1
            )
        except IndexError as e:
            self.log_error(
                message=f"IndexError during np.clip operation at idx={idx}.",
                episode=episode,
                step=step
            )
            self.new_group_structure = new_state
            return self.state.copy(), -100000, True, {}


        # Run OpenMC and get new group structure results: 
        keff_ngroup, rr_ngroup = self.compute_collapse(new_state)

        # Compute reward
        delta_keff = abs(self.ce_keff - keff_ngroup)
        # Broadcast collapsed group structure RR results to compare to origional group structures RR
        rr_expanded = np.zeros(self.max_idx)


        for i in range(len(rr_ngroup)):
            start = new_state[i]
            end = new_state[i+1]
            rr_expanded[start:end] = rr_ngroup[i]
        
        delta_rr = np.linalg.norm(self.ce_rr - rr_expanded)

        reward = ((100000 * delta_keff) + delta_rr)
        done = False
        # Add reward for delta_keff being less then uncertainty of CE keff
        ce_keff_std = self.ce_keff.std_dev
        if delta_keff <= ce_keff_std: 
            reward += 100
            
        if reward == -100000: 
            delta_keff = 0
            delta_rr=0

        self.steps_taken += 1
        self.state = new_state.copy()
        self.delta_keff = delta_keff
        self.delta_rr = delta_rr
        if self.steps_taken >= self.max_steps:
            return new_state.copy(), reward, True, {"info": "max steps reached"}
        
        return new_state.copy(), reward, done, {}

    def decode_action(self, index):
        # Decode a single integer action into boundary index, direction, and magnitude
        boundary_idx = index // (self.num_mags * self.num_dirs) + 1  # skip 0th and last
        rem = index % (self.num_mags * self.num_dirs)
        direction = 1 if (rem // self.num_mags) == 0 else -1
        magnitude = (rem % self.num_mags) + 1
        return boundary_idx, direction, magnitude

    def simulate_collapse(self, group_structure):
        # Placeholder for real collapse + keff / RR calculations
        fake_keff = self.ce_keff + np.random.normal(0, 0.001)
        fake_rr = self.ce_rr + np.random.normal(0, 0.01, size=self.ce_rr.shape)
        return fake_keff, fake_rr
    
    def compute_collapse(self, new_group_structure):
        sp = openmc.StatePoint(self.ce_sp, autolink=False)
        su = openmc.Summary(self.ce_summary)
        sp.link_with_summary(su)
        mgxs_lib = mgxsBuilder.create_mgxs_lib(self.group_boundaries_2k, self.geometry,self.tallies_file)

        mgxs_lib.load_from_statepoint(sp)
        # Find the names of all of the materials in our run
        all_materials_names = [mat.name for mat in self.geometry.get_all_materials().values()]

        # Initialize MGXS Library with OpenMC statepoint data
        mgxs_file = mgxs_lib.create_mg_library(xs_type='macro', xsdata_names=all_materials_names)
        # Save 200G version of mgxs file for later use
        mgxs_file.export_to_hdf5(self.output_path+'/mgxs_ce.h5')
        # Create a MGXS File which can then be written to disk
        #save mgxs 5 Group version
        collapsedGS = self.group_boundaries_2k[new_group_structure]
        groups = mgxs.EnergyGroups(group_edges=collapsedGS)

        mgxs_lib = mgxs_lib.get_condensed_library(groups)

        mgxs_file = mgxs_lib.create_mg_library(xs_type='macro', xsdata_names=all_materials_names)

        # Write the file to disk using the default filename of "mgxs.h5" and move to output folder
        mgxs_file.export_to_hdf5(self.output_path+'/mgxs.h5')
        # Re-define our materials to use the multi-group macroscopic data instead of the continuous-energy data.
        materials_mg, tallies_fileMG = mgxsBuilder.mgxs_materials_build(self.materials, self.output_path, new_group_structure)

        self.settings.export_to_xml(self.output_path)
        materials_mg.export_to_xml(self.output_path)
        tallies_fileMG.export_to_xml(self.output_path)

        openmc.run(cwd=self.output_path)

        # Read in results of run and return them 
        mg_spfile = os.path.join(self.output_path, 'statepoint_mg.h5')
        os.rename(self.output_path + '/statepoint.' + str(self.batches) + '.h5', mg_spfile)

        mg_sp = openmc.StatePoint(mg_spfile, autolink=False)
        # Get keff for collapsed group run
        mg_keff = mg_sp.keff

        tally = mg_sp.get_tally(name="absorption_tally")
        tally_array = tally.mean
        uncertainty = tally.std_dev
            
        num_materials = tally.filters[1].num_bins
        num_energy_bins = tally.filters[0].num_bins
            
        tally_per_bin = tally_array.reshape((num_energy_bins, num_materials))
        uncertainty_per_bin = uncertainty.reshape((num_energy_bins, num_materials))

        tally200G = np.sqrt((tally_per_bin**2).sum(axis=1))
        uncertainty_collapsed_group =uncertainty_per_bin.sum(axis=1)

        deltaE = np.diff(new_group_structure)
        mg_rr =tally200G/deltaE
        uncertainty_collapsed_group /= deltaE
        
        self.new_group_structure = new_group_structure
        self.mg_keff = mg_keff
        self.mg_rr = mg_rr
        return mg_keff, mg_rr
    
    @property
    def action_space(self):
        return self.action_dim

    @property
    def observation_space(self):
        return self.n_groups + 1
