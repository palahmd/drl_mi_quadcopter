from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import itm_quadcopter
from raisimGymTorch.algo.reinforcement_learning.ppo import PPO
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.helper.env_helper.env_helper import helper
import raisimGymTorch.algo.shared_modules.actor_critic as module
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
Runner file for training and retraining with PPO
"""

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
file_name = ""
if len(os.path.basename(__file__).split("_", 1)) != 1:
    for i in range(len(os.path.basename(__file__).split("_", 1)) - 1):
        file_name += os.path.basename(__file__).split("_", 1)[0] + "_"  # for dagger_runner.py -> file_name = dagger_
home_path = os.path.dirname(os.path.realpath(__file__)) + "/../.."
task_path = os.path.dirname(os.path.realpath(__file__))
raisim_unity_Path = home_path + "/raisimUnity/raisimUnity.x86_64"

# logging
saver = ConfigurationSaver(log_dir=home_path + "/training/imitation_learning",
                           save_items=[task_path + "/dagger_cfg.yaml", task_path + "/Environment.hpp"])
start_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
#tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first update

# config
cfg = YAML().load(open(task_path + "/" + file_name + "cfg.yaml", 'r'))
cfg['record'] = 'no'
cfg['environment']['render'] = False  # train environment should not be rendered
eval_cfg = YAML().load(open(task_path + "/" + file_name + "cfg.yaml", 'r'))
deterministic_policy = cfg['architecture']['deterministic_policy']

# create environment from the configuration file
env = VecEnv(itm_quadcopter.RaisimGymEnv(home_path + "/../rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], normalize_ob=False)
eval_env = VecEnv(
    itm_quadcopter.RaisimGymEnv(home_path + "/../rsc", dump(eval_cfg['environment'], Dumper=RoundTripDumper)),
    eval_cfg['environment'], normalize_ob=False)
env.turn_off_visualization()
eval_env.turn_off_visualization()

# action and observation space. Learner has 4 values less (quaternions)
ob_dim_expert = env.num_obs # expert has 4 additional values for quaternions
ob_dim_learner = ob_dim_expert - 4
act_dim = env.num_acts
obs = np.zeros((env.num_envs, ob_dim_learner), dtype="float32")

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
ppo = PPO(actor=actor, critic=critic,
          num_envs=cfg['environment']['num_envs'],
          num_transitions_per_env=n_steps,
          num_learning_epochs=cfg['hyperparam']['num_learning_epochs'],
          gamma=cfg['hyperparam']['Gamma'],
          lam=cfg['hyperparam']['Lambda'],
          num_mini_batches=cfg['hyperparam']['num_mini_batches'],
          device=device,
          log_dir=saver.data_dir,
          shuffle_batch=cfg['hyperparam']['shuffle'],
          learning_rate=cfg['hyperparam']['learning_rate'],
          value_loss_coef=cfg['hyperparam']['value_loss_coef'],
          use_clipped_value_loss=cfg['hyperparam']['use_clipped_value_loss'],
          entropy_coef=cfg['hyperparam']['entropy_coef'],
          deterministic_policy=cfg['architecture']['deterministic_policy'])

helper = helper(env=env, num_obs=ob_dim_learner,
                normalize_ob=cfg['helper']['normalize_ob'],
                update_mean=cfg['helper']['update_mean'],
                clip_action=cfg['helper']['clip_action'],
                scale_action=cfg['helper']['scale_action'])


if mode == 'retrain':
    helper.load_param(weight_path, actor, critic, ppo.optimizer, saver.data_dir, file_name)
    last_update = int(weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0])
    helper.load_scaling(weight_path, last_update)
else:
    last_update = 0

""" 
Training Loop
"""


for update in range(1000):
    env.reset()
    start = time.time()
    reward_sum = 0
    dones_sum = 0
    average_dones = 0.

    # optional: skip first visualization with update = 1
    if update == 0:
        update -= 1
        skip_eval = True
    update += last_update

    """ Evaluation and saving of the models """
    if update % cfg['environment']['eval_every_n'] == 0:
        print("Saving the current policy")
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
            #'scheduler_state_dict': learner.scheduler.state_dict(),
        }, saver.data_dir + "/full_" + str(update) + '.pt')
        helper.save_scaling(saver.data_dir, str(update))

        if cfg['environment']['visualize_eval']:
            print("Visualizing and evaluating the current policy")

            # open raisimUnity and wait until it has started and focused on robot
            proc = subprocess.Popen(raisim_unity_Path)
            eval_env.turn_on_visualization()
            for i in range(10): # reset 10 times to make sure that target point is changed
                eval_env.reset()
            time.sleep(6)
            eval_env.start_video_recording(start_date + "policy_" + str(update) + '.mp4')
            time.sleep(2)

            # load another graph to evaluate on the evaluation environment, so the loop inside the training environment
            # will not be falsified by the evaluation
            loaded_graph = module.MLP(cfg['architecture']['policy_net'], eval(cfg['architecture']['activation_fn']),
                                      ob_dim_learner, act_dim)
            loaded_graph.load_state_dict(
                torch.load(saver.data_dir + "/full_" + str(update) + '.pt')['actor_architecture_state_dict'])

            for step in range(int(n_steps * 1.5)):
                # separate and expert obs with dim 21 and (normalized) learner obs with dim 18
                full_obs = eval_env.observe()
                obs[0] = full_obs[0][0:18].copy()
                obs = helper.normalize_observation(obs)

                # limit action either by clipping or scaling
                action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu())
                action_ll = helper.limit_action(action_ll)

                reward, dones = eval_env.step(action_ll)

                # stop simulation for fluent visualization
                dones_sum += sum(dones)
                reward_sum += sum(reward)

                time.sleep(cfg['environment']['control_dt'])
            eval_env.stop_video_recording()
            eval_env.turn_off_visualization()

            # close raisimUnity to use less ressources
            proc.kill()
            # os.kill(proc.pid+1, signal.SIGKILL) # if proc.kill() does not work

            print('----------------------------------------------------')
            print('{:<40} {:>6}'.format("average dones: ", '{:0.6f}'.format(dones_sum)))
            print('{:<40} {:>6}'.format("average reward: ", '{:0.6f}'.format(reward_sum)))
            print('----------------------------------------------------\n')

    # actual training
    for step in range(n_steps):
        #env.turn_on_visualization()

        full_obs = env.observe()
        for i in range(0, env.num_envs):
            obs[i] = full_obs[i][0:18].copy()
        obs = helper.normalize_observation(obs)

        action = ppo.observe(obs)
        action = helper.limit_action(action)

        reward, dones = env.step(action)

        ppo.step(value_obs=obs, rews=reward, dones=dones)
        dones_sum = dones_sum + sum(dones)
        reward_sum = reward_sum + sum(reward)

        #env.turn_on_visualization()

    # take st step to get value obs
    full_obs = env.observe()
    for i in range(0, env.num_envs):
        obs[i] = full_obs[i][0:18].copy()
    obs = helper.normalize_observation(obs)

    if skip_eval:
        update += 1
        skip_eval = False

    mean_loss = ppo.update(actor_obs=obs,
               value_obs=obs,
               log_this_iteration=update % 10 == 0,
               update=update)

    end = time.time()

    average_ll_performance = reward_sum / total_steps
    average_dones = dones_sum / total_steps
    avg_rewards.append(average_ll_performance)

    actor.distribution.enforce_minimum_std((torch.ones(4)).to(device))

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("total reward: ", '{:0.10f}'.format(reward_sum)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("mean loss: ", '{:0.6f}'.format(mean_loss)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('action std: ')
    print(actor.distribution.std.cpu().detach().numpy())
    print('----------------------------------------------------\n')

