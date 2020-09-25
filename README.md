# Deep Reinforcement Learning of a quadcopter with model-indications - using raisim v 1.0

## Installation

Install the raisim library according to the description of www.raisim.com:

```commandline 
cd $WHERE_RAISIMLIB_IS_CLONED && mkdir build && cd build 
cmake .. -DCMAKE_INSTALL_PREFIX=~/raisim_install -DRAISIM_EXAMPLES=ON -DRAISIM_PY=ON && make install
``` 
 
## Setup and run

copy your activation key into the folder /rsc. It is blacklisted by .gitignore and will not be uploaded into this repository in a commit.

```commandline 
cd $WHERE_THIS_REPOSITORY_IS_CLONED && mkdir build && cd build 
cmake .. -DCMAKE_PREFIX_PATH=~/raisim_install && make 
``` 


