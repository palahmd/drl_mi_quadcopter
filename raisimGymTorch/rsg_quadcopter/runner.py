from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_quadcopter
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime

# directories
home_path = os.path.dirname(os.path.realpath(__file__)) + "/.."
env_path = os.path.dirname(os.path.realpath(__file__))

# config
cfg = YAML().load(open(env_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
env = VecEnv(rsg_quadcopter.RaisimGymEnv(home_path + "/../rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

# save the configuration and other files
#saver = ConfigurationSaver(log_dir=home_path + "/data/pidControl",
                          # save_items=[env_path + "/cfg.yaml", env_path + "/Environment.hpp"])

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs


for update in range(1000000):
    start = time.time()
    env.reset()
    action = np.arrange(0)

    env.turn_on_visualization()

    time.sleep(0.01)

    # actual training
    for step in range(n_steps):
        obs = env.observe()
        #action = ppo.observe(obs)
        rew, act = env.step(action)


    end = time.time()

    env.turn_off_visualization()


    #actor.distribution.enforce_minimum_std((torch.ones(12)*0.2).to('cuda'))

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    #print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    #print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    #print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    #print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('std: ')
    #print(np.exp(actor.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')
