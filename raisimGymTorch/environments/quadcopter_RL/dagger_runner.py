from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import quadcopter_RL
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.algo.pid_controller.pid_controller import PID
from raisimGymTorch.algo.imitation_learning.DAgger import DAgger
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
import subprocess, signal

"""
Initialization
Here:   - Observation space normalized
        - loss function: neg. log probability
"""

# configuration: mode, weight path and round number for beta in DAgger-Learner
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or retrain', type=str, default='train')
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
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../.."
raisim_unity_Path = home_path + "/raisimUnity/raisimUnity.x86_64"

# logging
saver = ConfigurationSaver(log_dir=home_path + "/training/imitation_learning",
                           save_items=[task_path + "/dagger_cfg.yaml", task_path + "/Environment.hpp"])
start_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
tensorboard_launcher(saver.data_dir + "/..")  # press refresh (F5) after the first ppo update

# config and config related options
cfg = YAML().load(open(task_path + "/" + file_name + "cfg.yaml", 'r'))
cfg['record'] = 'no'
cfg['environment']['render'] = False  # train environment should not be rendered
eval_cfg = YAML().load(open(task_path + "/" + file_name + "cfg.yaml", 'r'))
eval_cfg['environment']['num_envs'] = 1

# create environment and evaluation environment from the configuration file
env = VecEnv(quadcopter_RL.RaisimGymEnv(home_path + "/../rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], normalize_ob=False)
eval_env = VecEnv(
    quadcopter_RL.RaisimGymEnv(home_path + "/../rsc", dump(eval_cfg['environment'], Dumper=RoundTripDumper)),
    eval_cfg['environment'], normalize_ob=False)

# observation and action dim
ob_dim_expert = env.num_obs  # expert has 4 additional values for quaternions
ob_dim_learner = ob_dim_expert - 4
act_dim = env.num_acts
obs = np.zeros((env.num_envs, ob_dim_learner), dtype="float32")

# Training param
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

# Expert: PID Controller, target point and initial state to calculate target point
expert = PID(3, 50, 6.5, ob_dim_expert, act_dim, cfg['environment']['control_dt'], 1.727)
expert_actions = np.zeros(shape=(env.num_envs, act_dim), dtype="float32")
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

if mode == 'retrain':
    last_update = int(weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0])
else:
    last_update = 0

# Imitation Learner: DAgger
learner = DAgger(actor=actor, critic=critic, act_dim=act_dim,
                 num_envs=cfg['environment']['num_envs'],
                 num_transitions_per_env=n_steps,
                 num_mini_batches=cfg['hyperparam']['num_mini_batches'],
                 num_learning_epochs=cfg['hyperparam']['num_learning_epochs'],
                 log_dir=saver.data_dir,
                 beta=cfg['hyperparam']['Beta'],
                 l2_reg_weight=cfg['hyperparam']['l2_reg_weight'],
                 entropy_weight=cfg['hyperparam']['entropy_weight'],
                 use_lr_scheduler=cfg['hyperparam']['use_lr_scheduler'],
                 min_lr=cfg['hyperparam']['min_lr'],
                 max_lr=cfg['hyperparam']['max_lr'],
                 last_update=last_update,
                 beta_scheduler=cfg['hyperparam']['beta_scheduler'],
                 deterministic_policy=cfg['architecture']['deterministic_policy'],
                 device=device)

helper = helper(env=env, num_obs=ob_dim_learner,
                normalize_ob=cfg['helper']['normalize_ob'],
                update_mean=cfg['helper']['update_mean'],
                clip_action=cfg['helper']['clip_action'],
                scale_action=cfg['helper']['scale_action'])

learner.beta = cfg['hyperparam']['init_beta']

if mode == 'retrain':
    helper.load_param(weight_path, actor, critic, learner.optimizer, learner.scheduler, saver.data_dir, file_name)
    helper.load_scaling(weight_path, last_update)
    learner.beta = cfg['hyperparam']['init_beta'] - learner.beta_scheduler * last_update

""" 
Training Loop
"""

for update in range(2000):
    env.reset()
    start = time.time()

    # tmeps
    loopCount = 1
    dones_sum = 0
    reward_sum = 0

    # optional: skip first visualization with update = 1
    if update == 0:
        update = 0
        env.turn_off_visualization()
    update += last_update

    """adjust single visualizable env target to all envs. 
    Why this way? -> writing another python wrapper takes too much effort. The visualized env[0] creates a random 
    target, which is extracted via a C++->Python interface - in this case env.observe(). To apply the random target to 
    all envs, the target will be given to the otehr envs as an action with the only Python->C++ interface 
    env.step(action). In the Environment.hpp file, the in the step method the action (here: target of env[0]) can be set 
    as the target of all envs."""
    """
    expert_obs = env.observe()
    vis_target = init_state - expert_obs[0]
    for i in range(len(targets)):
        targets[i] = vis_target
        env_target[i] = vis_target[0][0:4]

    _, _ = env.step(env_target)
    expert_obs = env.observe() 
    """

    last_targets = targets.copy()
    expert_obs = env.observe()
    targets = init_state - expert_obs.copy()

    if last_targets[0][0] != targets[0][0]:
        print("new target")

    """ Evaluation and saving of the models """
    if update % cfg['environment']['eval_every_n'] == 0:
        print("Saving the current policy")
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': learner.optimizer.state_dict(),
            'scheduler_state_dict': learner.scheduler.state_dict(),
        }, saver.data_dir + "/full_" + str(update) + '.pt')
        helper.save_scaling(saver.data_dir, str(update))

        if cfg['environment']['visualize_eval']:
            print("Visualizing and evaluating the current policy")

            # open raisimUnity and wait until it has started and focused on robot
            proc = subprocess.Popen(raisim_unity_Path)
            eval_env.turn_on_visualization()
            for i in range(10):
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
                learner_obs = eval_env.observe()
                obs[0] = learner_obs[0][0:18].copy()
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

    """ Actual training """

    for step in range(n_steps):
        # separate and expert obs with dim=21 and (normalized) learner obs with dim=18
        learner_obs = expert_obs.copy()
        expert_obs += targets
        for i in range(0, env.num_envs):
            obs[i] = learner_obs[i][0:18]
        obs = helper.normalize_observation(obs)

        # choose an expert action per environment
        for i in range(0, env.num_envs):
            expert_obs_env_i = expert_obs[i, :]
            expert_actions[i, :] = expert.control(obs=expert_obs_env_i.reshape((ob_dim_expert, 1)),
                                                  target=targets[i][0:12].reshape((12, 1)), loopCount=loopCount)

        # choose an expert action with beta-probability or learner action with (1-beta) probability
        actions = learner.observe(actor_obs=obs, expert_actions=expert_actions, env_helper=helper)

        reward, dones = env.step(actions)
        learner.step(obs=obs, rews=reward, dones=dones)

        dones_sum += sum(dones)
        reward_sum += sum(reward)

        # for outter pid-control loop running five times slower
        if loopCount == 5:
            loopCount = 0
        loopCount += 1

        expert_obs = env.observe()

    learner_obs = expert_obs.copy()
    for i in range(0, env.num_envs):
        obs[i] = learner_obs[i][0:18]
    obs = helper.normalize_observation(obs)

    mean_loss, mean_action_loss, mean_action_log_prob_loss, mean_value_loss = learner.update(obs=obs,
                                                                                             log_this_iteration=update % 10 == 0,
                                                                                             update=update)
    actor.distribution.enforce_minimum_std((torch.ones(4)).to(device))

    end = time.time()

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average dones: ", '{:0.6f}'.format(dones_sum)))
    print('{:<40} {:>6}'.format("average reward: ", '{:0.6f}'.format(reward_sum)))
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
