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
actions = np.zeros(shape=(env.num_envs, act_dim), dtype="float32")
targets = np.zeros(shape=(env.num_envs, ob_dim), dtype="float32")
init_state = np.zeros(shape=(cfg['environment']['num_envs'], ob_dim), dtype="float32")
for i in range(len(init_state)):
    init_state[i][2] = 0.135
    init_state[i][3] = 1
    init_state[i][7] = 1
    init_state[i][11] = 1
    init_state[i][18] = 1

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

pid = PID(2.5, 50, 6.5, ob_dim, act_dim, cfg['environment']['control_dt'], 1.727)

for update in range(1000000):
    for i in range(10):
        env.reset()
    env.turn_on_visualization()
    loopCount = 0
    time.sleep(0.5)
    obs = env.observe()
    targets = init_state - obs.copy()

    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.

    for step in range(n_steps):
        frame_start = time.time()
        obs += targets
        for i in range(0, env.num_envs):
            expert_obs_env_i = obs[i, :]
            actions[i, :] = pid.control(obs=expert_obs_env_i.reshape((ob_dim, 1)),
                                                  target=targets[i][0:12].reshape((12, 1)), loopCount=loopCount)

        reward_ll, dones = env.step(actions)
        reward_ll_sum += sum(reward_ll)
        done_sum += sum(dones)

        obs = env.observe()
        
        # frequency of outter PID acceleration controller
        if loopCount == 8:
            loopCount = 0
            if step >= n_steps/4:
                loopCount = 3
        loopCount += 1

        frame_end = time.time()

        wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)

    print('----------------------------------------------------')
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(reward_ll_sum)))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(done_sum)))
    print('----------------------------------------------------\n')
    start_step_id = step + 1
    reward_ll_sum = 0.0

env.turn_off_visualization()