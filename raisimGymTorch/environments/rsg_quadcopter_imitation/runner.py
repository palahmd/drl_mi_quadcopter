from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_quadcopter_imitation
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.algo.pid_controller.pid_controller import PID
from raisimGymTorch.algo.imitation.DAgger import DAgger
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.helper.env_helper.env_helper import normalize_action, normalize_observation
import raisimGymTorch.algo.imitation.module as module
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
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../.."
raisim_unity_Path = home_path + "/raisimUnity/raisimUnity.x86_64"

# logging
saver = ConfigurationSaver(log_dir=home_path + "/training/imitation",
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])
#tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
env = VecEnv(rsg_quadcopter_imitation.RaisimGymEnv(home_path + "/../rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], normalize_ob=False)

# observation and action dim
ob_dim_expert = env.num_obs # expert has 4 additional values for quaternions
ob_dim_learner = ob_dim_expert - 4
act_dim = env.num_acts

# Training param
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs
normalize_learner_obs = cfg['environment']['normalize_ob']
log_prob_loss = cfg['environment']['log_prob_loss']

# Expert: PID Controller and target point
expert = PID(2.5, 20, 6.3, ob_dim_expert, act_dim, cfg['environment']['control_dt'], 1.727, normalize_action=True)
expert_actions = np.zeros(shape=(cfg['environment']['num_envs'], act_dim), dtype="float32")

target_point = np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype="float32")
target_point_n_envs = np.zeros(shape=(cfg['environment']['num_envs'], ob_dim_learner), dtype="float32")
for i in range (0, cfg['environment']['num_envs']):
    target_point_n_envs[i, :] = target_point

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

# Imitation Learner: DAgger
learner = DAgger(actor=actor, critic=critic, act_dim=act_dim,
                 num_envs=cfg['environment']['num_envs'],
                 num_transitions_per_env=n_steps,
                 num_mini_batches=8,
                 num_learning_epochs=4,
                 log_dir=saver.data_dir,
                 beta=0.1,
                 l2_reg_weight=0.0,
                 entropy_weight=0.0,
                 learning_rate=0.0005,
                 beta_scheduler=0.005,
                 device=device)

if mode == 'retrain':
    load_param(weight_path, env, actor, critic, learner.optimizer, saver.data_dir)


""" 
Training Loop
"""


for update in range(1000):
    env.reset()
    start = time.time()

    # tmeps
    loopCount = 5
    done_sum = 0
    average_dones = 0

    #if update == 0:
     #   update =1

    """ Evaluation and saving of the models """
    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': learner.optimizer.state_dict(),
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

            # separate and expert obs with dim 22 and (normalized) learner obs with dim 18
            # TODO: WTH DID I DO? TARGET POINT - NORMALIZED STATE?!
            expert_ob = env.observe(update_mean=False)
            expert_ob_clipped = expert_ob.copy()
            expert_ob_clipped = normalize_observation(env, expert_ob_clipped, normalize_ob=normalize_learner_obs)
            expert_ob_clipped.resize((cfg['environment']['num_envs'], 18), refcheck=False)
            learner_ob = target_point - expert_ob_clipped

            action_ll = loaded_graph.architecture(torch.from_numpy(learner_ob).cpu())
            print(action_ll)
            action_ll = normalize_action(action_ll)


            _, _ = env.step(action_ll.cpu().detach().numpy())

            time.sleep(cfg['environment']['control_dt'])

        env.stop_video_recording()
        env.turn_off_visualization()

        # close raisimUnity to use less ressources
        os.kill(proc.pid+1, signal.SIGKILL)

        env.reset()
        env.save_scaling(saver.data_dir, str(update))


    """ Atual training """
    for step in range(n_steps):
        #env.turn_on_visualization()

        # separate and expert obs with dim 22 and (normalized) learner obs with dim 18
        expert_obs = env.observe(update_mean=False)
        expert_obs_clipped = expert_obs.copy()
        expert_obs_clipped = normalize_observation(env, expert_obs_clipped, normalize_ob=normalize_learner_obs)
        expert_obs_clipped.resize((cfg['environment']['num_envs'], 18), refcheck=False)
        learner_obs = target_point_n_envs - expert_obs_clipped

        for i in range(0, len(expert_obs)):
            expert_obs_env_i = expert_obs[i, :].copy()
            expert_actions[i, :] = expert.control(obs=expert_obs_env_i.reshape((22, 1)),
                                                  target=target_point[0:12].reshape((12, 1)), loopCount=loopCount)


        learner_actions = learner.observe(actor_obs=learner_obs, expert_actions=expert_actions)

        _, _ = env.step(learner_actions)
        learner.step(obs=learner_obs, rews=_, dones=_)

        # for outter control loop
        if loopCount == 5:
            loopCount = 0
        loopCount += 1
        #env.turn_off_visualization()

        #frame_end = time.time()
        #wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
        #if wait_time > 0.:
         #   time.sleep(wait_time)

    mean_action_loss, mean_action_log_prob_loss = learner.update(log_this_iteration=update % 10 == 0, update=update,
                                                                 log_prob_loss=log_prob_loss)
    actor.distribution.enforce_minimum_std((torch.ones(4)).to(device))

    end = time.time()

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("beta: ", '{:0.6f}'.format(learner.beta + learner.beta_scheduler)))
    print('{:<40} {:>6}'.format("mean action log prob loss: ", '{:0.6f}'.format(mean_action_log_prob_loss)))
    print('{:<40} {:>6}'.format("mean action loss: ", '{:0.6f}'.format(mean_action_loss)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('action std: ')
    print(actor.distribution.std.cpu().detach().numpy())
    print('----------------------------------------------------\n')