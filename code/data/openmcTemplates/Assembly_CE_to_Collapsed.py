#! coding:utf-8

"""
Bailey Oteri 
04/25/25
# Code to create CE and 2000 group OpenMC data 

REFERENCES: 1.) BEAVRS code: https://github.com/mit-crpg/BEAVRS/tree/master/models/openmc
            2.) BEAVRS extract pincell example: https://github.com/mit-crpg/BEAVRS/blob/master/models/openmc/extract-pin.ipynb
            3.) OpenMC mgxs example: https://docs.openmc.org/en/v0.12.1/examples/mgxs-part-ii.html
"""
# ============================================================================
# Initialization - Edit this to use! 
# ============================================================================

import os  # to create folder and environment variable
import sys
import openmc  # for the python API of OpenMC 
import numpy as np  # for mathematical tools
import openmc.mgxs as mgxs # for multigroup cross section mode

import matplotlib
matplotlib.use('TkAgg')  # Use a GUI backend
import matplotlib.pyplot as plt
plt.ion()

from openmcTemplates import geometryBuilder, mgxsBuilder,setTallies

sys.path.append("PATH_TO_BEAVRS_CODE")
import beavrs.builder
import beavrs.constants as c

# Set the path to the cross-sections data
library_path = "PATH_TO_ENDF_DATA /cross_sections.xml"
os.environ["OPENMC_CROSS_SECTIONS"] = library_path


# For creating output folder
output_directory = os.getcwd()
output_path_base = "output"

# ============================================================================
# Output File path 
# ============================================================================
# Ensure the output_directory exists and get the list of files
try:
    existing_files = [d for d in os.listdir(output_directory)
                     if os.path.isdir(os.path.join(output_directory, d)) and
                     d.startswith(output_path_base) and
                     d[len(output_path_base):].isdigit()]
except FileNotFoundError:
    print(f"output_directory {output_directory} does not exist. Please check the path.")
    existing_files = []
# Find the highest number in existing filenames
if existing_files:
    max_number = max(int(d[len(output_path_base):]) for d in existing_files)
else:
    max_number = 0
# Increment the number for the new filename
new_number = max_number + 1
output_path = os.path.join(output_directory, f"{output_path_base}{new_number}")
os.makedirs(output_path)

# ============================================================================
# BEAVRS Geometry Model Builder
# ============================================================================

# Instantiate a BEAVRS object from the mit-crpg/PWR_benchmarks repository
b = beavrs.builder.BEAVRS()

# Get all OpenMC Lattices in a Python list
all_latts = b.openmc_geometry.get_all_lattices()

# Find the 1.6% enriched fuel lattice w/o BAs
assembly_name = 'Fuel 1.6% enr instr no BAs'
for id, latt in all_latts.items():
    if latt.name == assembly_name:
        assembly = latt

# Create surface objects for our "root" cell"
lattice_sides = openmc.model.rectangular_prism(17*c.pinPitch, 17*c.pinPitch,
                                                   boundary_type='reflective')
min_z = openmc.ZPlane(z0=c.struct_LowestExtent, boundary_type='vacuum')
max_z = openmc.ZPlane(z0=c.struct_HighestExtent, boundary_type='vacuum')

# Create a "root" cell filled by the fuel assembly
root_cell = openmc.Cell(name='root cell',
                        fill=assembly,
                        region=lattice_sides & +min_z & -max_z
                       )

# Create a "root" universe with ID=0 and add the "root" cell to it
root_univ = openmc.Universe(name='root universe', cells=[root_cell])

# Create an OpenMC Geometry around root Universe
geometry = openmc.Geometry(root_univ)

# ============================================================================
# Materials
# ============================================================================
# Get a list of all OpenMC Materials
all_materials = geometry.get_all_materials()

# Create a MaterialsFile object
materials = openmc.Materials(all_materials.values())

# Add cross section data to materials for openmc-plotter to work
materials.cross_sections = library_path

# ============================================================================
# Settings
# ============================================================================

# Create a MaterialsFile object
settings = openmc.Settings()

# Set neutron settings, example (Ref 2) used batches = 150, inactive = 10, particles = 1000
# Increasing number of neutrons for mgxs mode
batches = 400
settings.batches = batches
settings.inactive = 200
settings.particles = 10000

settings.run_mode = "eigenvalue"
settings.temperature = {'method': 'interpolation'}

# Use a bounding box to define the starting source distribution
lower_left = [-17*c.pinPitch/2, -17*c.pinPitch/2, c.fuel_ActiveFuel_bot]
upper_right = [+17*c.pinPitch/2, +17*c.pinPitch/2, c.fuel_ActiveFuel_top]
settings.source = openmc.source.Source(
    openmc.stats.Box(lower_left, upper_right, only_fissionable=True))

# ============================================================================
# Group Definition
# ============================================================================

# Create 2000 g and 20 g structures equal lethargy from 1E-8 ev to 10 MeV
twoKGroup = np.logspace(-8,7,num=2001)
indices = np.linspace(0, len(twoKGroup) - 1, 21, dtype=int)
collapsedGS = twoKGroup[indices]

# ============================================================================
# MGXS and Tallies files
# ============================================================================

# Initialize Tallies 
tallies_file = setTallies.create_tallies(materials,twoKGroup)
#create the mgxs Library
mgxs_lib = mgxsBuilder.create_mgxs_lib(twoKGroup, geometry,tallies_file)

# ============================================================================
# Create exports 
# ============================================================================
tallies_file.export_to_xml(output_path+'/tallies.xml')
geometry.export_to_xml(output_path+'/geometry.xml')
settings.export_to_xml(output_path+'/settings.xml')
materials.export_to_xml(output_path+'/materials.xml')
# ============================================================================
# Run OpenMC
# ============================================================================

openmc.run(cwd=output_path)

# ============================================================================
# Saving Exports From CE Run
# ============================================================================

# Move the statepoint Filemgxs_
ce_spfile = os.path.join(output_path, 'statepoint_ce.h5')
os.rename(output_path + '/statepoint.' + str(batches) + '.h5', ce_spfile)

# Move the Summary file
ce_sumfile = os.path.join(output_path, './summary_ce.h5')
os.rename(output_path + '/summary.h5', ce_sumfile)

# Load the statepoint file
sp = openmc.StatePoint(ce_spfile, autolink=False)

# Load the summary file in its new location
su = openmc.Summary(ce_sumfile)
sp.link_with_summary(su)

mgxs_lib.load_from_statepoint(sp)

# Find the names of all of the materials in our run
all_materials_names = [mat.name for mat in geometry.get_all_materials().values()]

# Initialize MGXS Library with OpenMC statepoint data
mgxs_file = mgxs_lib.create_mg_library(xs_type='macro', xsdata_names=all_materials_names)

settings.energy_mode = 'multi-group'

# Save 2000G version of mgxs file for later use
mgxs_file.export_to_hdf5(output_path+'/mgxs_ce.h5')
os.rename(output_path + '/tallies.xml', output_path+'/tallies_ce.xml')

# ============================================================================
# MultiGroup collapse xs
# ============================================================================

# Create a MGXS File which can then be written to disk

#save mgxs 20 Group version

groups = mgxs.EnergyGroups(group_edges=collapsedGS)

mgxs_lib = mgxs_lib.get_condensed_library(groups)

mgxs_file = mgxs_lib.create_mg_library(xs_type='macro', xsdata_names=all_materials_names)

# Write the file to disk using the default filename of "mgxs.h5" and move to output folder
mgxs_file.export_to_hdf5(output_path+'/mgxs.h5')

# ============================================================================
# Redefine Materials settings and tallies for openMG
# ============================================================================
# Re-define our materials to use the multi-group macroscopic data instead of the continuous-energy data.
materials_mg, tallies_fileMG = mgxsBuilder.mgxs_materials_build(materials, output_path, collapsedGS)

#create new mgxs_library with collapsed GS
mgxs_libMG = mgxsBuilder.create_mgxs_lib(collapsedGS, geometry,tallies_fileMG)

# ============================================================================
# Export to XML
# ============================================================================

# Export to "materials.xml"
settings.export_to_xml(output_path)
materials_mg.export_to_xml(output_path)
tallies_fileMG.export_to_xml(output_path)

# ============================================================================
# MultiGroup Run equla lethargy 20 groups
# ============================================================================

openmc.run(cwd=output_path)

# ============================================================================
# Save outputs
# ============================================================================

# Move the StatePoint File
mg_spfile = os.path.join(output_path, 'statepoint_mg.h5')
os.rename(output_path + '/statepoint.' + str(batches) + '.h5', mg_spfile)
# Move the Summary file
mg_sumfile = os.path.join(output_path, 'summary_mg.h5')
os.rename(output_path + '/summary.' + 'h5', mg_sumfile)

os.rename(output_path + '/tallies.xml', output_path+'/tallies_mg.xml')

# Rename and then load the last statepoint file and keff value
mgsp = openmc.StatePoint(mg_spfile, autolink=False)

# Load the summary file in its new location
mgsu = openmc.Summary(mg_sumfile)
mgsp.link_with_summary(mgsu)

mgxs_fileMG = mgxs_lib.create_mg_library(xs_type='macro', xsdata_names=all_materials_names)

mgxs_fileMG.export_to_hdf5(output_path+'/mgxs_mg.h5')
