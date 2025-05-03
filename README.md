# MGO_DQN_CSCI_575_Project
Code and instructions for Bailey Oteri's CSCI 575 final project - Multigroup Optimization using DQN RL.
Instructions for running code will be in the folder of the script. 
# Initialization Steps
## 1. ) Create Conda enviorment
I highly reccomend using a conda enviorment to dowload everything so it is easy to delete afterwards.
```
conda create --name MGO-DQN
conda activate MGO-DQN
```

## 2. ) Once in your conda enviorment, in command line download OpenMC.
OpenMC is an open source monte-carlo nuclear photon and neutron transport code.
For more information about the program itself, please go to: https://openmc.org/
To download in conda enviorment:
```
conda config --add channels conda-forge
conda install mamba
mamba install openmc
```
## 3.) Download Pytorch
The RL library I used was Pytorch, and my code was specifically written to use CUDA (NVIDA GPUs), however should also work on a CPU if CUDA is not avalible. 
If using an NVIDIA GPU, enter this command in your terminal: 
```
nvidia-smi
```
If you have the correct drivers installed for your GPU, the output should look something like this: 
![nvidia-smi](https://github.com/user-attachments/assets/aad410b7-f711-421e-aa7f-59f16298a788)

In the top right corner, you should see "CUDA Version: X". On the website linked above for getting started locally, input your CUDA Version and any other specifications you need to change, and the command you would need to run will be shown in order to download Pytorch. 

![Pytorch_Download_Command](https://github.com/user-attachments/assets/e4860376-13b0-46c4-b24d-42f1136686c5)

If downloading into a fresh conda enviorment, you should not need to worry about Python version or anything as those things will be set automatically.
### Check Download of Pytorch - Quickstart Tutorial
To check that Pytorch downloaded correctly and is using NVIDIA CUDA software, open the Quickstart.py folder and run in Python in the Pytorch conda enviorment. 
If everything downloaded properly, you should not need to make any edits to this code to run it! 
Comments are added throughout the code to breifly explain what is happening, for more details you can go to the following link for the full tutorial: 
```
https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
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
