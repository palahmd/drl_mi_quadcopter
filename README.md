# Deep Reinforcement Learning Control of a Quadcopter with model-indications - using raisim v 1.0
This repository belongs to the study Thesis "Deep Reinforcement Learning Control of a Quadcopter with model-indications".
The Quadcopter is first trained by an Imitation Learning algorithm with an PID-Controller as an expert to 
train an initial policy efficiently. Afterwards it is trained by an Reinforcement Learning algorithm
in different tasks to stabilize the control behaviour. 

## Getting Started
To run this Repository, you need to get a license and install RaiSim (www.raisim.com).
Install the RaiSim library according to the description of the website.

### Dependencies
* Eigen3: `sudo apt-get install libeigen3-dev` 
* cmake > 3.10
* vulkan, minizip and ffmpeg: 
  * `sudo apt install minizip ffmpeg`
  * vulkan: https://linuxconfig.org/install-and-test-vulkan-on-linux 
    
### Setup and run RaiSim
```commandline 
git clone https://github.com/raisimTech/raisimLib.git
cd raisimLib 
mkdir build && cd build 
cmake .. -DCMAKE_INSTALL_PREFIX=~/raisim_install -DRAISIM_EXAMPLES=ON -DRAISIM_PY=ON 
make install -j4
```
copy your activation key into the folder /rsc. It is blacklisted by .gitignore and will not be uploaded into this repository in a commit.




