from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import itm_quadcopter
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import os
import math
import time
from raisimGymTorch.algo.pid_controller.pid_controller import PID
import numpy as np
import torch

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
file_name = ""
if len(os.path.basename(__file__).split("_", 1)) != 1:
    for i in range(len(os.path.basename(__file__).split("_", 1)) - 1):
        file_name += os.path.basename(__file__).split("_", 1)[0] + "_"
home_path = os.path.dirname(os.path.realpath(__file__)) + "/../.."
task_path = os.path.dirname(os.path.realpath(__file__))

# config
cfg = YAML().load(open(task_path + "/" + file_name + "cfg.yaml", 'r'))

# create environment from the configuration file
env = VecEnv(itm_quadcopter.RaisimGymEnv(home_path + "/../rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], normalize_ob=False)

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts
#target_point = np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((12, 1))
target_point = np.zeros(shape=(1, ob_dim), dtype="float32")
init_state = np.zeros(shape=(1, ob_dim), dtype="float32")
init_state[0][2] = 0.135
init_state[0][3] = 1
init_state[0][7] = 1
init_state[0][11] = 1
init_state[0][18] = 1

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

pid = PID(2.8, 50, 6.5, ob_dim, act_dim, cfg['environment']['control_dt'], 1.727)

for update in range(1000000):
    env.reset()
    env.turn_on_visualization()
    loopCount = 8
    time.sleep(0.5)
    obs = env.observe()
    target_point = init_state - obs

    for step in range(n_steps):
        frame_start = time.time()
        obs += target_point
        action = pid.control(obs=obs.reshape((22, 1)), target=target_point[0][0:12].reshape(12,1), loopCount=loopCount)

        _, _ = env.step(action)
        obs = env.observe()
        
        # frequency of outter PID acceleration controller
        if loopCount >= 8:
            loopCount = 3
        loopCount += 1

        frame_end = time.time()

        wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)

env.turn_off_visualization()
