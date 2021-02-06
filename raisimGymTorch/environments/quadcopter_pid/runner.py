from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_quadcopter_pid
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
home_path = os.path.dirname(os.path.realpath(__file__)) + "/../.."
task_path = os.path.dirname(os.path.realpath(__file__))

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
env = VecEnv(rsg_quadcopter_pid.RaisimGymEnv(home_path + "/../rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], normalize_ob=False)

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts
target_point = np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((12, 1))

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

pid = PID(2, 10, 6, ob_dim, act_dim, cfg['environment']['control_dt'], 1.727)

for update in range(1000000):
    env.reset()
    env.turn_on_visualization()
    loopCount = 5

    for step in range(n_steps):
        frame_start = time.time()
        obs = env.observe(update_mean=True)

        action = pid.smallAnglesControl(obs=obs.reshape((22, 1)), target=target_point, loopCount=loopCount)

        _, _ = env.step(action)

        if loopCount == 5:
            loopCount = 0
        loopCount += 1

        frame_end = time.time()

        wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)

env.turn_off_visualization()
