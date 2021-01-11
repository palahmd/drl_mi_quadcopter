from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_quadcopter_pid
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
home_path = os.path.dirname(os.path.realpath(__file__)) + "/../.."
env_path = os.path.dirname(os.path.realpath(__file__))

print(home_path)

# config
cfg = YAML().load(open(env_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
env = VecEnv(rsg_quadcopter_pid.RaisimGymEnv(home_path + "/../rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# save the configuration and other files
saver = ConfigurationSaver(log_dir=home_path + "/data/pidControl",
                          save_items=[env_path + "/cfg.yaml", env_path + "/Environment.hpp"])

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'],
                                        nn.LeakyReLU,
                                        ob_dim,
                                        act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0),
                         device)

critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'],
                                          nn.LeakyReLU,
                                          ob_dim,
                                          1),
                           device)

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.996,
              lam=0.95,
              num_mini_batches=4,
              device=device,
              log_dir=saver.data_dir,
              mini_batch_sampling='in_order',
              )


for update in range(1000000):
    start = time.time()
    env.reset()
    action = []
    env.turn_on_visualization()
    time.sleep(0.01)

    # actual training
    for step in range(n_steps):
        obs = env.observe()
        action = ppo.observe(obs)
        reward, dones = env.step(action)


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
