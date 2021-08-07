from ruamel.yaml import YAML, dump, RoundTripDumper
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
import subprocess

"""
This script trains or retrains an agent with the PPO-method. The configuration is specified in the cfg-file.
    * example for running a training in environment of stage 1:
        python ppo_runner.py -e stage_1
    * example for running a retraining of an existing agent in environment of stage 2:
        python ppo_runner.py -m retrain -w $PATH_TO_THE_NEURAL_NETWORK_MODEL.pt -e stage_2
"""

# configuration: train a new agent or retrain an existing agent with:
# python ppo_runner.py -m retrain -w $PATH_TO_THE_NEURAL_NETWORK_MODEL.pt -e stage_2
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
parser.add_argument('-e', '--environment', help='set environment', type=str, default='stage_2')
args = parser.parse_args()
mode = args.mode
weight_path = args.weight
env_mode = args.environment

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
file_name = ""
if len(os.path.basename(__file__).split("_", 1)) != 1:
    for i in range(len(os.path.basename(__file__).split("_", 1)) - 1):
        file_name += os.path.basename(__file__).split("_", 1)[0] + "_"  # for ppo_runner.py -> file_name = ppo_
home_path = os.path.dirname(os.path.realpath(__file__)) + "/../.."
task_path = os.path.dirname(os.path.realpath(__file__))
raisim_unity_Path = home_path + "/raisimUnity/raisimUnity.x86_64"

# config
cfg = YAML().load(open(task_path + "/" + file_name + "cfg.yaml", 'r'))

# create environment from the configuration file and save configuration files
if env_mode == 'stage_1':
    from raisimGymTorch.env.bin import stage_1_target_tracking
    env = VecEnv(stage_1_target_tracking.RaisimGymEnv(home_path + "/../rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
                 cfg['environment'], normalize_ob=False)

    saver = ConfigurationSaver(log_dir=home_path + "/training/ppo",
                               save_items=[task_path + "/" + file_name + "cfg.yaml", home_path + "/environments/stage_1_target_tracking/Environment.hpp"])
elif env_mode == 'stage_2':
    from raisimGymTorch.env.bin import stage_2_state_recovery
    env = VecEnv(stage_2_state_recovery.RaisimGymEnv(home_path + "/../rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
                 cfg['environment'], normalize_ob=False)

    saver = ConfigurationSaver(log_dir=home_path + "/training/ppo",
                               save_items=[task_path + "/" + file_name + "cfg.yaml", home_path + "/environments/stage_2_state_recovery/Environment.hpp"])
else:
    raise Exception("Make sure to select a correct environment. Take a look at the runner-script")

# logging
start_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first update

# action and observation space. Learner has 4 values less (expert PID-feedback)
ob_dim_expert = env.num_obs
ob_dim_learner = ob_dim_expert - 4
act_dim = env.num_acts
obs = np.zeros((env.num_envs, ob_dim_learner), dtype="float32")

# training steps
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

# actor and critic
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

# PPO-learner
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
          l2_reg_coef=cfg['hyperparam']['l2_reg_coef'],
          bc_coef=cfg['hyperparam']['bc_coef'],
          deterministic_policy=cfg['architecture']['deterministic_policy'])

# environment helper for specific operations
helper = helper(env=env, num_obs=ob_dim_learner,
                normalize_ob=cfg['helper']['normalize_ob'],
                update_mean=cfg['helper']['update_mean'],
                clip_action=cfg['helper']['clip_action'],
                scale_action=cfg['helper']['scale_action'])

if mode == 'retrain':

    helper.load_param(weight_path, actor, critic, ppo, saver.data_dir, file_name)
    last_update = int(weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0])
    helper.load_scaling(weight_path, last_update)
    # actor.distribution.enforce_minimum_std((torch.ones(4)*0.4).to(device))
else:
    last_update = 0
    # obs_var is determined analytically/experimentally and rounded. Helps the neural network to better approx. actions,
    # observation is assumed to be nearly gaussian distributed
    obs_var = np.array([10/np.sqrt(3), 10/np.sqrt(3), 10/np.sqrt(3),
                        np.sqrt(9), np.sqrt(9), np.sqrt(9), np.sqrt(9), np.sqrt(9), np.sqrt(9), np.sqrt(9), np.sqrt(9), np.sqrt(9),
                        5, 5, 5,
                        5, 5, 5], dtype="float32")
    for i in range(env.num_envs):
        helper.obs_rms.var[i] = obs_var


""" 
Training Loop
"""

for update in range(3000):
    env.reset()
    start = time.time()

    #temps
    reward_sum = 0
    done_sum = 0

    # optional: skip first evaluation with skip_eval = True
    if update == 0:
        skip_eval = False
    update += last_update

    """ Evaluation and saving of the models """
    if update % cfg['environment']['vis_every_n'] == 0 and skip_eval == False:
        print("Saving the current policy")

        # save neural network model of actor and critic
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
            #'scheduler_state_dict': ppo.scheduler.state_dict(),
        }, saver.data_dir + "/full_" + str(update) + '.pt')
        helper.save_scaling(saver.data_dir, str(update))

        # if visualize_eval = True in the config, the current agent will be evaluated and visualized in the current
        # environment.
        if cfg['environment']['visualize_eval']:
            print("Visualizing and evaluating the current policy")

            # open raisimUnity, wait until it has started and focused on robot
            proc = subprocess.Popen(raisim_unity_Path)
            env.turn_on_visualization()
            time.sleep(6)
            env.start_video_recording(start_date + "policy_" + str(update) + '.mp4')
            time.sleep(2)

            # to find out in which/how many environments the agent failed
            done_vec = np.zeros(shape=(int(n_steps * 1.5), cfg["environment"]["num_envs"], 1), dtype="bool")

            for step in range(int(n_steps * 1.5)):
                # separate expert obs with dim 22 and (normalized) ppo obs with dim 18
                full_obs = env.observe()
                [obs, expert_actions] = np.hsplit(full_obs, [18])
                obs = helper.normalize_observation(obs)

                # limit action either by clipping or scaling
                action = ppo.actor.noiseless_action(torch.from_numpy(obs).to(device))
                action = helper.limit_action(action)

                reward, dones = env.step(action)
                done_vec[step] = dones.reshape(env.num_envs, 1).copy()

                done_sum += sum(dones)
                reward_sum += sum(reward)

                # slow down visualization
                time.sleep(cfg['environment']['control_dt'])

            env.stop_video_recording()
            env.turn_off_visualization()
            for i in range(9): # reset 9 times to make sure that target point is changed for the new training iteration
                env.reset()

            # close raisimUnity to use less resources
            proc.kill()

            num_failed_envs, _ = helper.identify_failed_envs(done_vec)

            print('----------------------------------------------------')
            print('{:<40} {:>6}'.format("total dones: ", '{:0.6f}'.format(done_sum)))
            print('{:<40} {:>6}'.format("failed environments: ", '{:0.6f}'.format(num_failed_envs)))
            print('{:<40} {:>6}'.format("total reward: ", '{:0.6f}'.format(reward_sum)))
            print('----------------------------------------------------\n')

            done_sum = 0
            reward_sum = 0

    """ Actual training """
    for step in range(n_steps):
        # get the observation vector from the environment
        full_obs = env.observe()
        [obs, expert_actions] = np.hsplit(full_obs, [18])
        obs = helper.normalize_observation(obs)

        # send in the observation vector and sample actions
        action = ppo.observe(expert_actions, obs)
        action = helper.limit_action(action)

        # perform the actions in the respective environments and get the rewards
        reward, dones = env.step(action)

        # add all transitions
        ppo.step(value_obs=obs, rews=reward, dones=dones)
        done_sum = done_sum + sum(dones)
        reward_sum = reward_sum + sum(reward)

    # update actor and critic
    full_obs = env.observe()
    [obs, expert_actions] = np.hsplit(full_obs, [18])
    obs = helper.normalize_observation(obs)

    mean_loss = ppo.update(actor_obs=obs,
               value_obs=obs,
               log_this_iteration=update % 10 == 0,
               update=update,
               reward_sum=reward_sum/env.num_envs)

    end = time.time()

    if skip_eval:
        skip_eval = False

    # enforce minimum standard deviation of the actor distribution to ancourage exploration
    #actor.distribution.enforce_minimum_std((torch.ones(4)*0.2).to(device))

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(reward_sum / total_steps)))
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

