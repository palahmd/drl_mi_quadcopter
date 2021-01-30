from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_quadcopter_ppo
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

print(home_path)

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
env = VecEnv(rsg_quadcopter_ppo.RaisimGymEnv(home_path + "/../rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts
goal_point = np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((12, 1))


# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs


pid = PID(2, 20, 6, ob_dim, act_dim, cfg['environment']['simulation_dt'], 1.727)



for update in range(1000000):
    start = time.time()
    env.reset()
    env.turn_on_visualization()
    loopCount = 5

    for step in range(n_steps):
        time.sleep(0.01)
        obs = env.observe()

        action = pid.smallAnglesControl(obs.reshape((18, 1)), goal_point, loopCount)
        print(action)

        _, _ = env.step(action)

        if loopCount == 5:
            loopCount = 0
        loopCount += 1


    env.turn_off_visualization()







