from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_quadcopter_imitation
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.algo.pid_controller.pid_controller import PID
from raisimGymTorch.algo.imitation.DAgger import DAgger
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
import raisimGymTorch.algo.imitation.module as module
import os
import math
import time
import numpy as np
import torch
import argparse
import torch.nn as nn

"""
Initialization
"""

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or retrain', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
args = parser.parse_args()
mode = args.mode
weight_path = args.weight

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
home_path = os.path.dirname(os.path.realpath(__file__)) + "/../.."
task_path = os.path.dirname(os.path.realpath(__file__))

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
env = VecEnv(rsg_quadcopter_imitation.RaisimGymEnv(home_path + "/../rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], normalize_ob=False)

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

# Set up PID Controller and target point
expert = PID(2, 10, 6, ob_dim, act_dim, cfg['environment']['control_dt'], 1.727, normalize_action=True)
target_point = np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype="float32").reshape((18, 1))

# Set up Actor Critic
actor = module.Actor(module.MLP(cfg['architecture']['policy_net'],
                                        nn.ReLU,
                                        ob_dim,
                                        act_dim),
                         module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0),
                         device)

critic = module.Critic(module.MLP(cfg['architecture']['value_net'],
                                          nn.ReLU,
                                          ob_dim,
                                          1),
                           device)

# Set up DAgger learner
if mode == 'retrain':
    # save the configuration and related files to pre-trained model
    if weight_path == "":
        raise Exception("\nCan't find the pre-trained weight, please provide a pre-trained weight with --weight switch\n")
    print("\nRetraining from the policy:", weight_path+".pt\n")

    full_checkpoint_path = weight_path.rsplit('/', 1)[0] + '/' + 'full_' + weight_path.rsplit('/', 1)[1].split('_', 1)[1] + '.pt'
    mean_csv_path = weight_path.rsplit('/', 1)[0] + '/' + 'mean' + weight_path.rsplit('/', 1)[1].split('_', 1)[1] + '.csv'
    var_csv_path = weight_path.rsplit('/', 1)[0] + '/' + 'var' + weight_path.rsplit('/', 1)[1].split('_', 1)[1] + '.csv'
    saver = ConfigurationSaver(log_dir=home_path + "/data/ppo",
                               save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"],
                               pretrained_items=[weight_path.rsplit('/', 1)[0].rsplit('/', 1)[1], [weight_path+'.pt', weight_path+'.txt', full_checkpoint_path, mean_csv_path, var_csv_path]])
    ## load observation scaling from files of pre-trained model
    env.load_scaling(weight_path.rsplit('/', 1)[0], int(weight_path.rsplit('/', 1)[1].split('_', 1)[1]))
    print("Load observation scaling in", weight_path.rsplit('/', 1)[0]+":", "mean"+str(int(weight_path.rsplit('/', 1)[1].split('_', 1)[1])) + ".csv", "and", "var"+str(int(weight_path.rsplit('/', 1)[1].split('_', 1)[1])) + ".csv")
    ## load actor and critic parameters from full checkpoint
    checkpoint = torch.load(full_checkpoint_path)
    actor.architecture.load_state_dict(checkpoint['actor_architecture_state_dict'])
    actor.distribution.load_state_dict(checkpoint['actor_distribution_state_dict'])
    critic.architecture.load_state_dict(checkpoint['critic_architecture_state_dict'])
else:
    # save the configuration and other files
    saver = ConfigurationSaver(log_dir=home_path + "/training/imitation",
                               save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])

learner = DAgger(actor=actor, critic=critic,
                 num_envs=cfg['environment']['num_envs'],
                 num_transitions_per_env=n_steps,
                 num_mini_batches=4,
                 num_learning_epochs=4,
                 beta=0.15,
                 l2_reg_weight=0.1,
                 device=device)

if mode == 'retrain':
    ## load optimizer parameters from full checkpoint
    learner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


"""
Training
"""

for update in range(1000000):
    env.reset()
    env.turn_on_visualization()
    loopCount = 5

    for step in range(n_steps):
        frame_start = time.time()
        expert_obs = env.observe(update_mean=True)
        learner_obs = target_point.reshape((1, 18)) - expert_obs

        expert_actions = expert.control(obs=expert_obs.reshape((18, 1)), target=target_point[0:12], loopCount=loopCount)
        learner_actions = learner.observe(actor_obs=learner_obs, expert_actions=expert_actions)

        rewards, dones = env.step(learner_actions)
        learner.step(value_obs=learner_obs, expert_actions=expert_actions, rews=rewards, dones=dones)
        print(learner_actions)


        if loopCount == 5:
            loopCount = 0
        loopCount += 1

        frame_end = time.time()

        wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)

    expert_obs = env.observe(update_mean=True)
    learner_obs = target_point.reshape((1, 18)) - expert_obs
    learner.update()

    actor.distribution.enforce_minimum_std((torch.ones(4)*0.2).to(device))

env.turn_off_visualization()