# MGO_DQN_CSCI_575_Project
Code and instructions for Bailey Oteri's CSCI 575 final project - Multigroup Optimization using DQN RL.

Tutorial pt 1 is just the following steps, and was too large to add to github because each step takes a few min to download all of the libraries. Tutorial pt 2 starts right after the following steps have been completed, and the github repository has been downloaded as a zip file. 

NOTE: when downloading the github repo, if planning to run full program, please go into the data folder and unzip 'statepoint_ce.h5', it was too large to upload without compressing first.

# Initialization Steps
## 1. ) Create Conda enviorment
I highly reccomend using a conda enviorment to dowload everything so it is easy to delete afterwards.
```
conda create --name MGO
conda activate MGO
```

## 2. ) Download OpenMC 
OpenMC is an open source monte-carlo nuclear photon and neutron transport code.
For more information about the program itself, please go to: https://openmc.org/

To download in conda enviorment:
```
conda config --add channels conda-forge
conda install mamba
mamba install openmc
```
## 3.) Download Stable Baselines3
For this project, I used Stable Baselines3's library. To download it and its dependancies in your conda enviorment, enter this command: 
```
pip install stable-baselines3[extra]
```

## 4.) Optional - Download ENDF cross section library
In order to use OpenMC in continuous energy mode, you need the appropriate microscopic cross section libraries. To run this project as is, this is not required, it is only required if trying to run the code to build the BEAVRS assembly (or any other OpenMC run).
To download the correct one, please go to : 
'''
https://openmc.org/official-data-libraries/
'''
Scroll down to ENDF/B-VII.1 and download the .tar.xz file listed. Once downloaded and unzipped, you will need the path to the the "cross_sections.xml" file for running OpenMC. 
## 5.) Optional - Download BEAVRS repo
For this project, I used the BEAVRS reactor, a IRPhE 2023 benchmark draft. 
Information and source code can be found at: 
```
https://github.com/mit-crpg/BEAVRS
```
Downloading this code is not nessisary to run the project as-is using the statepoint HDF5 file I provided. However, if wanting to experement with any other configuration of the reactor, the source code will be nessisary. 
