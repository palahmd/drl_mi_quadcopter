from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import raisimGymTorch.algo.shared_modules.actor_critic as module
from raisimGymTorch.helper.env_helper.env_helper import helper
import os
import math
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import datetime

""" 
This script automatically tests all existing evaluation agents in a directory and creates csv-files for:
    * rewards for each environment and each agent
    * failed tasks for each environment and each agent
"""

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
parser.add_argument('-e', '--environment', help='set environment', type=str, default='stage_1')
args = parser.parse_args()
env_mode = args.environment

# directories
home_path = os.path.dirname(os.path.realpath(__file__)) + "/../.."
task_path = os.path.dirname(os.path.realpath(__file__))
weight_path = args.weight.rsplit('/', 1)[0]

# config: take the setup of the existing cfg-file
cfg_file = []
for file in os.listdir(weight_path):
    if file.endswith(".yaml"):
        cfg_file.append(file)

if len(cfg_file) > 1:
    raise Exception("Too many cfg-files in the directory, not sure which one to choose")

cfg = YAML().load(open(weight_path + "/" + cfg_file[0], 'r'))

# specify maximum time step
cfg['environment']['max_time'] = 15.0

# create environment from the configuration file
if env_mode == 'stage_1':
    from raisimGymTorch.env.bin import stage_1_testenv
    env = VecEnv(stage_1_testenv.RaisimGymEnv(home_path + "/../rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
                 cfg['environment'], normalize_ob=False)
elif env_mode == 'stage_2':
    from raisimGymTorch.env.bin import stage_2_testenv
    env = VecEnv(stage_2_testenv.RaisimGymEnv(home_path + "/../rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
                 cfg['environment'], normalize_ob=False)
else:
    raise Exception("Make sure to select a correct environment. Take a look at the tester-script.")

# shortcuts
ob_dim_expert = env.num_obs
ob_dim_learner = ob_dim_expert - 4
act_dim = env.num_acts
obs = np.zeros((env.num_envs, ob_dim_learner), dtype="float32")
targets = np.zeros((env.num_envs, 3), dtype="float32")

# environment helper scale_obs_rms=True will enhance preformance of the agent
helper = helper(env=env, num_obs=ob_dim_learner,
                normalize_ob=cfg['helper']['normalize_ob'],
                update_mean=cfg['helper']['update_mean'],
                clip_action=cfg['helper']['clip_action'],
                scale_action=cfg['helper']['scale_action'],
                scale_obs_rms=True)

iteration_number = int(args.weight.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0])

helper.load_scaling(args.weight, iteration_number)


if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))
    start = time.time()

    env.turn_on_visualization()
    time.sleep(3)
    env.reset()

    # steps
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # temps for output csv-file
    all_agent_rewards = []
    all_agent_dones = []

    # list all agents
    agent_list = []
    for file in os.listdir(weight_path):
        if file.endswith(".pt"):
            agent_list.append(int(file.split("_", 1)[1].split(".", 1)[0]))
    agent_list.sort()

    max_iteration = agent_list[-1]
    min_iteration = agent_list[0]
    delta_iteration = agent_list[1]-agent_list[0]

    """ Actual testing loop """
    for i in range(int((max_iteration-min_iteration)/delta_iteration) +1):

        print('----------------------------------------------------')
        print('{:<40} {:>6}'.format("Agent: ", '{:0.10f}'.format(iteration_number + i * delta_iteration)))

        # build and load neural network with deterministic actions
        if cfg['architecture']['shared_nets']:
            loaded_graph = module.sharedBaseNetMLP(cfg['architecture']['base_net'], cfg['architecture']['policy_net'], cfg['architecture']['value_net'],
                                                   eval(cfg['architecture']['activation_fn']), ob_dim_learner, [act_dim,1])
            loaded_graph.load_state_dict(torch.load(args.weight.rsplit('/', 1)[0]+"/full_"+str(iteration_number + i * delta_iteration)+'.pt')['actor_architecture_state_dict'])
        else:
            loaded_graph = module.MLP(cfg['architecture']['policy_net'], eval(cfg['architecture']['activation_fn']), ob_dim_learner, act_dim)
            loaded_graph.load_state_dict(torch.load(args.weight.rsplit('/', 1)[0]+"/full_"+str(iteration_number + i * delta_iteration)+'.pt')['actor_architecture_state_dict'])

            env.turn_on_visualization()
            env.reset()

        # start recording
        #env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "test"+'.mp4')
        #time.sleep(2)

        # temps
        reward_sum= 0
        done_sum = 0
        done_vec = np.zeros(shape=(int(n_steps * 1.5), cfg["environment"]["num_envs"], 1), dtype="bool")
        current_index = []

        # logging of failed envs
        full_obs = env.observe()
        [targets, _] = np.hsplit(full_obs, [3])
        average_task_duration = 0
        task_finished = np.zeros(env.num_envs, dtype=bool)
        fin_count = 0
        path_length = 0

        """ Actual testing loop """
        for step in range(int(n_steps)):
            frame_start = time.time()
            [obs, expert_actions] = np.hsplit(full_obs, [18])
            eval_obs = obs.copy()

            obs = obs * (1 + np.random.normal(0,0.05))
            obs = helper.normalize_observation(obs)

            # check if target is reached within specific tolerance
            for i in range(env.num_envs):
                path_length += np.sqrt(np.power(eval_obs[i][12],2) + np.power(eval_obs[i][13],2) + np.power(eval_obs[i][14],2)) * cfg["environment"]["control_dt"]
                if np.sqrt(np.power(eval_obs[i][0],2) + np.power(eval_obs[i][1],2) + np.power(eval_obs[i][2],2)) < 0.1:
                    if task_finished[i] == False and step > 200:
                        average_task_duration += step
                        task_finished[i] = True
                        fin_count += 1

            if cfg['architecture']['shared_nets']:
                action = loaded_graph.actor_net(torch.from_numpy(obs))
            else:
                action = loaded_graph.architecture(torch.from_numpy(obs))
            action = helper.limit_action(action)

            reward, dones = env.step(action)
            reward_sum+= sum(reward)
            done_sum += sum(dones)
            done_vec[step] = dones.reshape(env.num_envs, 1).copy()

            # slow down visualization
            frame_end = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
            if wait_time > 0:
                time.sleep(wait_time)

            full_obs = env.observe()

        """ Statistics """

        num_failed_envs, index = helper.identify_failed_envs(done_vec)
        new_index = [env for env in index if env not in current_index]
        current_index = index
        if len(new_index) > 0:
            print('----------------------------------------------------')
            print('{:<40} {:>6}'.format("total reward: ", '{:0.10f}'.format(reward_sum)))
            print('{:<40} {:>6}'.format("dones: ", '{:0.10f}'.format(done_sum)))
            print('{:<40} {:>6}'.format("failed environments: ", '{:0.6f}'.format(num_failed_envs)))
            for i in range(len(new_index)):
                print("failed target:")
                print(targets[new_index[i]])
            print('----------------------------------------------------\n')
        new_index.clear()

        all_agent_rewards.append(reward_sum)
        all_agent_dones.append(num_failed_envs)

# export csv-files
np.savetxt(os.path.join(task_path + "/outputs", 'all_agents_rewards.csv'), all_agent_rewards, delimiter=",")
np.savetxt(os.path.join(task_path + "/outputs", 'all_agents_failed_envs.csv'), all_agent_dones, delimiter=",")

env.stop_video_recording()
env.turn_off_visualization()
print("task_finished at the maximum visualization steps")
