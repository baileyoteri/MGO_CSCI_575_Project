#! coding:utf-8

"""
Bailey Oteri & Josh Nichols
Spring 2025
Create a multi-group cross section library for OpenMC

Changes:    
    - 01/08/25: Initial version
    - 03/04/25: Added function to create materials for MGXS 

Inputs: None
Outputs: geometry and materials
"""
import openmc  # for the python API of OpenMC 
import openmc.mgxs as mgxs # for multigroup cross section mode
import xml.etree.ElementTree as ET
from . import setTallies

def create_mgxs_lib(groupEdges,geometry,tallies_file):

    groups = mgxs.EnergyGroups(group_edges=groupEdges)

    # Initialize a 2-group MGXS Library for OpenMC
    mgxs_lib = openmc.mgxs.Library(geometry)
    mgxs_lib.energy_groups = groups

    # Specify multi-group cross section types to compute
    mgxs_lib.mgxs_types = ['total', 'absorption', 'nu-fission', 'fission',
                       'nu-scatter matrix', 'multiplicity matrix', 'chi']
    # Specify a "cell" domain type for the cross section tally filters
    mgxs_lib.domain_type = "material"

    # Specify the cell domains over which to compute multi-group cross sections
    mgxs_lib.domains = geometry.get_all_materials().values()

    # Do not compute cross sections on a nuclide-by-nuclide basis
    mgxs_lib.by_nuclide = False

    # Set the Legendre order to 3 for P3 scattering
    mgxs_lib.legendre_order = 1

    # Check the library - if no errors are raised, then the library is satisfactory.
    mgxs_lib.check_library_for_openmc_mgxs()
    
    mgxs_lib.build_library()

    mgxs_lib.add_to_tallies_file(tallies_file, merge=True)

    return mgxs_lib

def mgxs_materials_build(materials, output_path, collapsedGS):

    # Re-define our materials to use the multi-group macroscopic data
    # instead of the continuous-energy data.

    # Create a list to store OpenMC material objects
    materials_list = []

    # Loop through materials and add macroscopic cross-sections
    for mat in materials:
        new_mat = openmc.Material(mat.id, name=mat.name)  # Create a new OpenMC material
        new_mat.add_macroscopic(mat.name)  # Add macroscopic cross-section using the material's name
        materials_list.append(new_mat)

    # Create OpenMC Materials object
    materials_mg = openmc.Materials(materials_list)
    materials_mg.cross_sections = output_path+'/mgxs.h5'

    #create new empty tallies file
    tallies_fileMG = setTallies.create_tallies(materials=materials_mg,energyGroups= collapsedGS)

    return materials_mg, tallies_fileMG
