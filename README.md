# Deep Reinforcement Learning-based Control for the Quadcopter with model-indications - using RaiSim v1.0.0
This repository is part of the study Thesis "Deep Reinforcement Learning-based Control for the Quadcopter with Model-Indications".
The DRL-based controller is trained in a two-staged training approach as illustrated in the Figure below. In the first training 
stage, a pre-training of the controller is performed with an Imitation Learning algorithm where an initial policy is trained with the Data Aggregation (DAgger) method. The supervised learning objective
is the control strategy of a PID-controller for small angle conditions on the given states. Thereafter, the policy of the 
DRL-based controller is further trained in a second training stage with Proximal Policy Optimization (PPO), where the controller is
supposed to explore a more robust control strategy and enhance stability in a more challenging task.

## Getting Started
To run the code, a license of RaiSim is needed. RaiSim provides a commercial license, a trial license 
and an academic license which can be requested on https://raisim.com/sections/License.html.
The following instructions contain the steps for a setup on Linux.
For the installation on Mac or Windows, follow the instructions on https://raisim.com/sections/Installation.html.

### Dependencies
Before setting up RaiSim, the following dependencies need to be installed. It is recommended to set up a virtual environment 
like a conda environment (https://docs.anaconda.com/).
* Python > 3.5 (3.8 or higher recommended)
* Eigen3: `sudo apt-get install libeigen3-dev` 
* cmake > 3.10
* vulkan, minizip and ffmpeg: 
  * `sudo apt install minizip ffmpeg`
  * vulkan: https://linuxconfig.org/install-and-test-vulkan-on-linux
* PyTorch and cuda: follow the instructions on https://pytorch.org/get-started/locally/
    
### Setup of RaiSim
Execute the following lines to download and setup RaiSim. It is recommended into clone the raisimLib repository to the home folder. 
```commandline 
git clone https://github.com/raisimTech/raisimLib.git
cd raisimLib 
mkdir build && cd build 
cmake .. -DCMAKE_INSTALL_PREFIX=~/raisim_install -DRAISIM_PY=ON 
make install -j4
```

### Setup of this Repository
Execute the following lines to download and setup this repository:
```commandline 
git clone https://github.com/Pala-Ah/drl_mi_quadcopter.git
cd drl_mi_quadcopter/raisimGym/scripts
bash init.sh
bash compile.sh
```
When executing `bash init.sh`, you will be asked to specify the directory to raisimLib. `init.sh` basically links
the algorithm and environment folders to the the raisimGymTorch module of raisimLib. `compile.sh` compiles the environments in
raisimGymTorch. Before running RaiSim, you also need to copy your activation key (license) into the folder /rsc. 
It is blacklisted by .gitignore and will not be uploaded into this repository in a commit.

## How to run the Scripts
The training and testing python-scripts are provided in the respective environment folders. The bash-files in the
raisimGym/scripts folder exemplify how to train, re-train or test an agent. 
### Running a Pre-Trained Agent
This repository provides two pre-trained agents: One from the first and one from the second training stage. For instance, 
to run the PPO-agent in 

## Folder Structure
* raisim: C++ version of the quadcopter controlled by a PID-controller. It needs to be built with cmake.
* raisimGym: DRL training environment and scripts for automated training
  * agents: Pre-trained agents.
  * algorithms: Code library containing the Neural Network module and the DAgger, PPO and PID-controller algorithms
  * environments: Different training and testing environments. The Environment.hpp file contains the simulation of the robot
  and the task setup. In some environments, it also contains the PID-controller which provides a control feedback in the training phase
  * helper: an enironment helper for some additional functions. 
  * scripts: Setup, training and testing scripts. The running and testing scripts show how to run the respective scripts with and without a 
    pre-trained neural network model.
* raisimPy: The python version of the quadcopter. It solely launches the quadcopter.
* rsc: contains the urdf model of the quadcopter and the activation key. 



