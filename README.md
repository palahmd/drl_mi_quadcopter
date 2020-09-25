# Deep Reinforcement Learning of a quadcopter with model-indications - using raisim v 1.0

## Installation

Install the raisim library according to the description of www.raisim.com:

```commandline 
cd $WHERE_RAISIMLIB_IS_CLONED && mkdir build && cd build 
cmake .. -DCMAKE_INSTALL_PATH=~/raisim_install -DRAISIM_EXAMPLES=ON -DRAISIM_PY=ON && make 
``` 
 
## Setup and run

```commandline 
cd $WHERE_THIS_REPOSITORY_IS_CLONED && mkdir build && cd build 
cmake .. -DCMAKE_PREFIX_PATH=~/raisim_install && make 
``` 


