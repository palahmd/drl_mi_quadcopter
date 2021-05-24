from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import itm_quadcopter_ppo
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
saver = ConfigurationSaver(log_dir=home_path + "/training/ppo",
                           save_items=[task_path + "/ppo_cfg.yaml", task_path + "/Environment.hpp"])
start_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first update

# config
cfg = YAML().load(open(task_path + "/" + file_name + "cfg.yaml", 'r'))
eval_cfg = YAML().load(open(task_path + "/" + file_name + "cfg.yaml", 'r'))
deterministic_policy = cfg['architecture']['deterministic_policy']

# create environment from the configuration file
env = VecEnv(itm_quadcopter_ppo.RaisimGymEnv(home_path + "/../rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], normalize_ob=False)
env.turn_off_visualization()

# action and observation space. Learner has 4 values less (quaternions)
ob_dim_expert = env.num_obs # expert has 4 additional values for quaternions
ob_dim_learner = ob_dim_expert - 4
act_dim = env.num_acts
obs = np.zeros((env.num_envs, ob_dim_learner), dtype="float32")
targets = np.zeros(shape=(env.num_envs, ob_dim_expert), dtype="float32")
vis_target = np.zeros(shape=(1, ob_dim_expert), dtype="float32")
env_target = np.zeros(shape=(env.num_envs, act_dim), dtype="float32")
init_state = np.zeros(shape=(cfg['environment']['num_envs'], ob_dim_expert), dtype="float32")
for i in range(len(init_state)):
    init_state[i][2] = 0.135
    init_state[i][3] = 1
    init_state[i][7] = 1
    init_state[i][11] = 1
    init_state[i][18] = 1

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
          clip_param=cfg["hyperparam"]["clip_param"],
          entropy_coef=cfg['hyperparam']['entropy_coef'],
          deterministic_policy=cfg['architecture']['deterministic_policy'])

helper = helper(env=env, num_obs=ob_dim_learner,
                normalize_ob=cfg['helper']['normalize_ob'],
                update_mean=cfg['helper']['update_mean'],
                clip_action=cfg['helper']['clip_action'],
                scale_action=cfg['helper']['scale_action'])


if mode == 'retrain':
    helper.load_param(weight_path, actor, critic, ppo, saver.data_dir, file_name)
    last_update = int(weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0])
    helper.load_scaling(weight_path, last_update)
else:
    last_update = 0
    obs_var = np.array([10/np.sqrt(3), 10/np.sqrt(3), 10/np.sqrt(3),
                        np.sqrt(9), np.sqrt(9), np.sqrt(9), np.sqrt(9), np.sqrt(9), np.sqrt(9), np.sqrt(9), np.sqrt(9), np.sqrt(9),
                        5, 5, 5,
                        5, 5, 5], dtype="float32")
    print(helper.obs_rms.var.shape)
    for i in range(env.num_envs):
        helper.obs_rms.var[i] = obs_var
    print(helper.obs_rms.var.shape)

actor.distribution.enforce_minimum_std((torch.ones(4)*0.4).to(device))

""" 
Training Loop
"""


for update in range(10000):
    env.reset()
    start = time.time()
    reward_sum = 0
    done_sum = 0
    average_dones = 0.

    # optional: skip first visualization with update = 1
    if update == 0:
        update -= 1
        skip_eval = True
    update += last_update

    """ Evaluation and saving of the models """
    if update % cfg['environment']['vis_every_n'] == 0:
        print("Saving the current policy")
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
            #'scheduler_state_dict': ppo.scheduler.state_dict(),
        }, saver.data_dir + "/full_" + str(update) + '.pt')
        helper.save_scaling(saver.data_dir, str(update))

        if cfg['environment']['visualize_eval']:
            print("Visualizing and evaluating the current policy")

            # open raisimUnity and wait until it has started and focused on robot
            proc = subprocess.Popen(raisim_unity_Path)
            env.turn_on_visualization()
            time.sleep(6)
            env.start_video_recording(start_date + "policy_" + str(update) + '.mp4')
            time.sleep(2)

            done_vec = np.zeros(shape=(int(n_steps * 1.5), cfg["environment"]["num_envs"], 1), dtype="bool")

            for step in range(int(n_steps * 1.5)):
                # separate and expert obs with dim 21 and (normalized) ppo obs with dim 18
                full_obs = env.observe()
                [obs, expert_actions] = np.hsplit(full_obs, [18])
                obs = helper.normalize_observation(obs)

                # limit action either by clipping or scaling
                action = ppo.actor.noiseless_action(torch.from_numpy(obs).to(device))
                action = helper.limit_action(action)

                reward, dones = env.step(action)
                done_vec[step] = dones.reshape(env.num_envs, 1).copy()

                # stop simulation for fluent visualization
                done_sum += sum(dones)
                reward_sum += sum(reward)

                time.sleep(cfg['environment']['control_dt'])

            env.stop_video_recording()
            env.turn_off_visualization()
            for i in range(9): # reset 9 times to make sure that target point is changed for new training iteration
                env.reset()

            # close raisimUnity to use less ressources
            proc.kill()
            # os.kill(proc.pid+1, signal.SIGKILL) # if proc.kill() does not work

            num_failed_envs, _ = helper.identify_failed_envs(done_vec)

            print('----------------------------------------------------')
            print('{:<40} {:>6}'.format("total dones: ", '{:0.6f}'.format(done_sum)))
            print('{:<40} {:>6}'.format("failed environments: ", '{:0.6f}'.format(num_failed_envs)))
            print('{:<40} {:>6}'.format("total reward: ", '{:0.6f}'.format(reward_sum)))
            print('----------------------------------------------------\n')

            done_sum = 0
            reward_sum = 0

    # actual training
    for step in range(n_steps):
        full_obs = env.observe()
        [obs, expert_actions] = np.hsplit(full_obs, [18])
        obs = helper.normalize_observation(obs)

        action = ppo.observe(expert_actions, obs)
        action = helper.limit_action(action)

        reward, dones = env.step(action)

        ppo.step(value_obs=obs, rews=reward, dones=dones)
        done_sum = done_sum + sum(dones)
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
               log_this_iteration=update % 5 == 0,
               update=update,
                reward_sum=reward_sum/env.num_envs)

    end = time.time()

    average_ll_performance = reward_sum / total_steps
    average_dones = done_sum / total_steps
    avg_rewards.append(average_ll_performance)

    #actor.distribution.enforce_minimum_std((torch.ones(4)*0.2).to(device))

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("total reward: ", '{:0.10f}'.format(reward_sum)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(done_sum)))
    print('{:<40} {:>6}'.format("mean loss: ", '{:0.6f}'.format(mean_loss)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('action std: ')
    print(actor.distribution.std.cpu().detach().numpy())
    print('----------------------------------------------------\n')

