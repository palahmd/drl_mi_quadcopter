from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_quadcopter_ppo
from raisimGymTorch.algo.ppo.ppo import PPO
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.helper.env_helper.env_helper import normalize_action, normalize_observation
import raisimGymTorch.algo.ppo.module as module
import os
import math
import time
import numpy as np
import torch
import argparse
import torch.nn as nn
import datetime
import subprocess, signal


"""
Initialization
Here:   - Observation space normalized
        - loss function: neg. log probability
"""


# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
args = parser.parse_args()
mode = args.mode
weight_path = args.weight

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
home_path = os.path.dirname(os.path.realpath(__file__)) + "/../.."
task_path = os.path.dirname(os.path.realpath(__file__))
raisim_unity_Path = home_path + "/raisimUnity/raisimUnity.x86_64"

# logging
saver = ConfigurationSaver(log_dir=home_path + "/training/imitation",
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])
#tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
env = VecEnv(rsg_quadcopter_ppo.RaisimGymEnv(home_path + "/../rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], normalize_ob=False)

# shortcuts
ob_dim_expert = env.num_obs # expert has 4 additional values for quaternions
ob_dim_learner = ob_dim_expert - 4
act_dim = env.num_acts
normalize_learner_obs = cfg['environment']['normalize_ob']

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

avg_rewards = []

# Actor and Critic
actor = module.Actor(module.MLP(cfg['architecture']['policy_net'],
                                eval(cfg['architecture']['activation_fn']),
                                ob_dim_learner,
                                act_dim),
                     module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0),
                     device=device)

critic = module.Critic(module.MLP(cfg['architecture']['value_net'],
                                  eval(cfg['architecture']['activation_fn']),
                                  ob_dim_learner,
                                  1),
                       device=device)

# PPO Learner
ppo = PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.996,
              lam=0.95,
              num_mini_batches=4,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              )

if mode == 'retrain':
    load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)

for update in range(1000):
    start = time.time()
    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.

    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')

        # we create another graph just to demonstrate the save/load method
        loaded_graph = module.MLP(cfg['architecture']['policy_net'], eval(cfg['architecture']['activation_fn']), ob_dim_learner, act_dim)
        loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

        # open raisimUnity and wait until it has started and focused on robot
        proc = subprocess.Popen(raisim_unity_Path)
        time.sleep(5)
        env.turn_on_visualization()
        env.reset()
        time.sleep(2)
        env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')
        time.sleep(2)

        for step in range(int(n_steps*1.5)):
            expert_ob = env.observe(update_mean=False)
            expert_ob_clipped = expert_ob.copy()
            #expert_ob_clipped = normalize_observation(env, expert_ob_clipped, normalize_ob=normalize_learner_obs)
            expert_ob_clipped.resize((cfg['environment']['num_envs'], 18), refcheck=False)
            obs = expert_ob_clipped

            action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu())
            action_ll = normalize_action(action_ll)

            reward_ll, dones = env.step(action_ll.cpu().detach().numpy())

            time.sleep(cfg['environment']['control_dt'])

        env.stop_video_recording()
        env.turn_off_visualization()

        # close raisimUnity to use less ressources
        os.kill(proc.pid+1, signal.SIGKILL)

        env.reset()
        env.save_scaling(saver.data_dir, str(update))

    # actual training
    for step in range(n_steps):
        #env.turn_on_visualization()

        expert_ob = env.observe(update_mean=True)
        expert_ob_clipped = expert_ob.copy()
        expert_ob_clipped = normalize_observation(env, expert_ob_clipped, normalize_ob=normalize_learner_obs)
        expert_ob_clipped.resize((cfg['environment']['num_envs'], 18), refcheck=False)
        obs = expert_ob_clipped

        action = ppo.observe(obs)
        action = normalize_action(torch.from_numpy(action).to(device))

        reward, dones = env.step(action)

        ppo.step(value_obs=obs, rews=reward, dones=dones)
        done_sum = done_sum + sum(dones)
        reward_ll_sum = reward_ll_sum + sum(reward)

    # take st step to get value obs
    expert_ob = env.observe(update_mean=True)
    expert_ob_clipped = expert_ob.copy()
    expert_ob_clipped = normalize_observation(env, expert_ob_clipped, normalize_ob=normalize_learner_obs)
    expert_ob_clipped.resize((cfg['environment']['num_envs'], 18), refcheck=False)
    obs = expert_ob_clipped
    ppo.update(actor_obs=obs,
               value_obs=obs,
               log_this_iteration=update % 10 == 0,
               update=update)

    end = time.time()

    average_ll_performance = reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    avg_rewards.append(average_ll_performance)

    actor.distribution.enforce_minimum_std((torch.ones(4)*0.2).to(device))

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('std: ')
    print(np.exp(actor.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')

