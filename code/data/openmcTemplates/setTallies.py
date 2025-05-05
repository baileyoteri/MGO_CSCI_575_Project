#! coding:utf-8

"""
Bailey Oteri 
01/08/25
Tallies for continuous energy and multigroup energy for OpenMC

Inputs: materials, energyGroups, filters=['energy','material'],pitch=1.25984
TODO: make mesh based on geometry file dimensions. 
Outputs: Tallies
"""

import numpy as np
import openmc

def create_tallies(materials, energyGroups, filters=['energy','material'],pitch=1.25984,dim=10):

    # Initialize the tallies file
    tallies_file = openmc.Tallies()
    filterObjects = {}
    # Continuous-Energy (CE) Energy Filter
    filterObjects['energy'] = openmc.EnergyFilter(energyGroups)

    # Particle Filter
    filterObjects['particle'] = openmc.ParticleFilter(["neutron"])

    # Material Filter (makes geometry tallies instead of soup)
    filterObjects['material'] = openmc.MaterialFilter(materials)

    # Mesh Filter (vertical slices)
    # Instantiate a tally Mesh
    mesh = openmc.RegularMesh()
    mesh.type = 'regular'
    mesh.dimension = [dim, dim]
    mesh.lower_left = [-pitch/2, -pitch/2]
    mesh.upper_right = [+pitch/2, +pitch/2]

    # Instantiate tally Filter
    filterObjects['mesh'] = openmc.MeshFilter(mesh)

    # Nuclides for heating tally
    nuclides = ["U238", "H1", "U235", "O16", "B10"]

    # Continuous-Energy Tallies
    fluxTally = openmc.Tally(name='flux_tally')
    fluxTally.scores = ['flux']
    fluxTally.filters = [filterObjects[flt] for flt in filters if flt in filterObjects]
    tallies_file.append(fluxTally)

    absorptionTally = openmc.Tally(name='absorption_tally')
    absorptionTally.scores = ['absorption']
    absorptionTally.filters = [filterObjects[flt] for flt in filters if flt in filterObjects]
    tallies_file.append(absorptionTally)

    fissionTally = openmc.Tally(name='fission_tally')
    fissionTally.scores = ['fission']
    fissionTally.filters = [filterObjects[flt] for flt in filters if flt in filterObjects]
    tallies_file.append(fissionTally)

    return tallies_file
