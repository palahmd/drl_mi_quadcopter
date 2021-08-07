from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.algo.imitation_learning.dagger import DAgger
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, tensorboard_launcher
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
This script trains or retrains an agent with the DAgger-method. The configuration is specified in the cfg-file.
    * example for running a training in environment of stage 1:
        python dagger_runner.py -e stage_1
    * example for running a retraining of an existing agent in environment of stage 2:
        python dagger_runner.py -m retrain -w $PATH_TO_THE_NEURAL_NETWORK_MODEL.pt -e stage_2
"""

# configuration: train a new agent or retrain an existing agent with:
# python dagger_runner.py -m retrain -w $PATH_TO_THE_NEURAL_NETWORK_MODEL.pt
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or retrain', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
parser.add_argument('-e', '--environment', help='set environment', type=str, default='stage_1')
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
        file_name += os.path.basename(__file__).split("_", 1)[0] + "_"  # for dagger_runner.py -> file_name = dagger_
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../.."
raisim_unity_Path = home_path + "/raisimUnity/raisimUnity.x86_64"

# config
cfg = YAML().load(open(task_path + "/" + file_name + "cfg.yaml", 'r'))

# create environment from the configuration file and save configuration files
if env_mode == 'stage_1':
    from raisimGymTorch.env.bin import stage_1_target_tracking
    env = VecEnv(stage_1_target_tracking.RaisimGymEnv(home_path + "/../rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
                 cfg['environment'], normalize_ob=False)

    saver = ConfigurationSaver(log_dir=home_path + "/training/dagger",
                               save_items=[task_path + "/" + file_name + "cfg.yaml", home_path + "/environments/stage_1_target_tracking/Environment.hpp"])
elif env_mode == 'stage_2':
    from raisimGymTorch.env.bin import stage_2_state_recovery
    env = VecEnv(stage_2_state_recovery.RaisimGymEnv(home_path + "/../rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
                 cfg['environment'], normalize_ob=False)

    saver = ConfigurationSaver(log_dir=home_path + "/training/dagger",
                               save_items=[task_path + "/" + file_name + "cfg.yaml", home_path + "/environments/stage_2_state_recovery/Environment.hpp"])
else:
    raise Exception("Make sure to select a correct environment. Take a look at the runner-script")

# logging
start_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first update

# observation and action dim
ob_dim_expert = env.num_obs  # expert has 4 additional values for quaternions
ob_dim_learner = ob_dim_expert - 4
act_dim = env.num_acts
obs = np.zeros((env.num_envs, ob_dim_learner), dtype="float32")

# training param
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs
last_checkpoint = 0 # for recreating neural network params from last checkpoint

# Expert PID controller: target point and initial state to calculate target point
expert_actions = np.zeros(shape=(env.num_envs, act_dim), dtype="float32")

# Actor and Critic
if cfg['architecture']['shared_nets']:
    actor_critic_module = module.sharedBaseNetMLP(cfg['architecture']['base_net'], cfg['architecture']['policy_net'],
                                                  cfg['architecture']['value_net'],
                                                  eval(cfg['architecture']['activation_fn']), ob_dim_learner,
                                                  [act_dim, 1])
    actor_module = actor_critic_module
    critic_module = actor_critic_module
else:
    actor_module = module.MLP(cfg['architecture']['policy_net'], eval(cfg['architecture']['activation_fn']),
                              ob_dim_learner, act_dim)
    critic_module = module.MLP(cfg['architecture']['policy_net'], eval(cfg['architecture']['activation_fn']),
                               ob_dim_learner, 1)

actor = module.Actor(actor_module, module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0), device=device,
                     shared_nets=cfg['architecture']['shared_nets'])

critic = module.Critic(critic_module, device=device, shared_nets=cfg['architecture']['shared_nets'])

# Imitation Learner: DAgger
if mode == 'retrain':
    last_update = int(weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0])
else:
    last_update = 0

learner = DAgger(actor=actor, critic=critic, act_dim=act_dim,
                 num_envs=cfg['environment']['num_envs'],
                 num_transitions_per_env=n_steps,
                 num_mini_batches=cfg['hyperparam']['num_mini_batches'],
                 num_learning_epochs=cfg['hyperparam']['num_learning_epochs'],
                 log_dir=saver.data_dir,
                 beta=cfg['hyperparam']['beta_min'],
                 gamma=cfg['hyperparam']['gamma'],
                 lam=cfg['hyperparam']['lam'],
                 l2_reg_weight=cfg['hyperparam']['l2_reg_weight'],
                 entropy_weight=cfg['hyperparam']['entropy_weight'],
                 use_lr_scheduler=cfg['hyperparam']['use_lr_scheduler'],
                 min_lr=cfg['hyperparam']['min_lr'],
                 max_lr=cfg['hyperparam']['max_lr'],
                 last_update=last_update,
                 beta_scheduler=cfg['hyperparam']['beta_scheduler'],
                 deterministic_policy=cfg['architecture']['deterministic_policy'],
                 shuffle_batch=cfg['hyperparam']['shuffle'],
                 device=device)

learner.beta = cfg['hyperparam']['init_beta']

# environment helper for specific operations
helper = helper(env=env, num_obs=ob_dim_learner,
                normalize_ob=cfg['helper']['normalize_ob'],
                update_mean=cfg['helper']['update_mean'],
                clip_action=cfg['helper']['clip_action'],
                scale_action=cfg['helper']['scale_action'])

if mode == 'retrain':
    # loading network, optimizer, scheduler and observation scaling params
    helper.load_param(weight_path, actor, critic, learner, saver.data_dir, file_name)
    helper.load_scaling(weight_path, last_update)
    #actor.distribution.enforce_minimum_std((torch.ones(4)*0.4).to(device))

    """ to adjust beta probability of learner according to update number. Perhaps last if-loop needs to be fixed
    last_cfg = YAML().load(open(weight_path.rsplit('/', 1)[0] + '/' + file_name + "cfg.yaml", 'r'))
    delta_beta = cfg['hyperparam']['init_beta'] - cfg['hyperparam']['beta_min']
    last_delta_beta = last_cfg['hyperparam']['init_beta'] - last_cfg['hyperparam']['beta_min']
    last_beta_mod = round(last_cfg['hyperparam']['beta_scheduler'] * last_update,4) % (last_delta_beta)
    if last_beta_mod < delta_beta: # beta decreasing
        learner.beta = last_cfg['hyperparam']['init_beta'] - (last_delta_beta - last_beta_mod)
    else: # beta increasing
        learner.beta = last_cfg['hyperparam']['init_beta'] - (2*last_delta_beta - last_beta_mod)
        learner.beta_scheduler = -abs(learner.beta_scheduler)
    """

else:
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

for update in range(2000):
    env.reset()
    start = time.time()

    # tmeps
    done_sum = 0
    reward_sum = 0

    """ Evaluation and Visualization"""
    # optional: skip first visualization with update = -1
    if update == 0:
        skip_eval = False
    update += last_update - last_checkpoint

    """ Evaluation and saving of the models """
    if update % cfg['environment']['vis_every_n'] == 0 and skip_eval == False:
        print("Saving the current policy")

        # save neural network model of actor and critic
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': learner.optimizer.state_dict(),
            #'scheduler_state_dict': learner.scheduler.state_dict(),
        }, saver.data_dir + "/full_" + str(update) + '.pt')
        helper.save_scaling(saver.data_dir, str(update))

        # if visualize_eval = True in the config, the current agent will be evaluated and visualized in the current
        # environment.
        if cfg['environment']['visualize_eval']:
            print("Visualizing and evaluating the current policy")

            # open raisimUnity and wait until it has started and focused on robot
            proc = subprocess.Popen(raisim_unity_Path)
            env.turn_on_visualization()
            time.sleep(6)
            env.start_video_recording(start_date + "policy_" + str(update) + '.mp4')
            time.sleep(2)

            # to find out in which/how many environments the agent failed
            done_vec = np.zeros(shape=(int(n_steps * 1.5), cfg["environment"]["num_envs"], 1), dtype="bool")

            for step in range(int(n_steps * 1.5)):
                # separate and expert obs with dim 22 and (normalized) learner obs with dim 18
                full_obs = env.observe()
                [obs, expert_actions] = np.hsplit(full_obs, [18])
                obs = helper.normalize_observation(obs)

                # limit action either by clipping or scaling
                action = learner.actor.noiseless_action(torch.from_numpy(obs).to(device))
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

            # close raisimUnity to use less ressources
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

        # choose an expert action with beta-probability or learner action with (1-beta) probability
        actions = learner.observe(actor_obs=obs, expert_actions=expert_actions, env_helper=helper)

        # perform the actions in the respective environments and get the rewards
        reward, dones = env.step(actions)

        # add all transitions
        learner.step(rews=reward, dones=dones)
        done_sum += sum(dones)
        reward_sum += sum(reward)

    """ a method to prevent the learner from learning from failed tasks. The environments where the learner failed will 
        be sorted out and replaced randomly with trajectories of successful environments. 
    if done_sum != 0:
        failed_envs = learner.storage.filter_failed_envs()
        print('----------------------------------------------------')
        print("controller failed, failed trajectories purged")
        print('{:<40} {:>6}'.format("failed environments: ", '{:0.6f}'.format(failed_envs)))
        print('----------------------------------------------------\n')
    """

    # update actor and critic
    full_obs = env.observe()
    [obs, expert_actions] = np.hsplit(full_obs, [18])
    obs = helper.normalize_observation(obs)
    mean_loss, mean_action_loss, mean_action_log_prob_loss, mean_value_loss = learner.update(obs=obs,
                                                                                             log_this_iteration=update % 1 == 0,
                                                                                             update=update,
                                                                                             mean_reward=reward_sum/env.num_envs)

    tot_dones = learner.tot_dones
    failed_envs = learner.failed_envs

    if skip_eval:
        skip_eval = False

    end = time.time()

    # enforce minimum standard deviation of the actor distribution to ancourage exploration
    #actor.distribution.enforce_minimum_std((torch.ones(4)*0.2).to(device))

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("tot dones in simulation: ", '{:0.6f}'.format(done_sum)))
    print('{:<40} {:>6}'.format("tot dones in train step: ", '{:0.6f}'.format(tot_dones)))
    print('{:<40} {:>6}'.format("failed_envs: ", '{:0.6f}'.format(failed_envs)))
    print('{:<40} {:>6}'.format("total reward: ", '{:0.6f}'.format(reward_sum)))
    print('{:<40} {:>6}'.format("beta: ", '{:0.6f}'.format(learner.beta + learner.beta_scheduler)))
    print('{:<40} {:>6}'.format("mean loss: ", '{:0.6f}'.format(mean_loss)))
    print('{:<40} {:>6}'.format("mean action log prob loss: ", '{:0.6f}'.format(mean_action_log_prob_loss)))
    print('{:<40} {:>6}'.format("mean action loss: ", '{:0.6f}'.format(mean_action_loss)))
    print('{:<40} {:>6}'.format("mean value loss: ", '{:0.6f}'.format(mean_value_loss)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('action std: ')
    print(actor.distribution.std.cpu().detach().numpy())
    print('----------------------------------------------------\n')

    """ a method to prevent the learner from learning from failed tasks. the params of the last checkpoint will be
        recreated. 
    if done_sum =! 0:
        last_checkpoint += helper.restart_from_last_checkpoint(env=env, saver=saver, actor=actor, critic=critic,
                                                              learner=learner, update_num=update)
        print('----------------------------------------------------')
        print('{:<40} {:>6}'.format("Controller failed, last checkpoint restored to: ", '{:0.6f}'.format(update - last_checkpoint)))
        print('----------------------------------------------------\n')
    """
