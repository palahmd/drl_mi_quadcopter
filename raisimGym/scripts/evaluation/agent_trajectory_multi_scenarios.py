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
This script tests an agent on a task and outputs a csv-file with the trajectory
"""

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
parser.add_argument('-e', '--environment', help='set environment', type=str, default='target_sequence')
args = parser.parse_args()
env_mode = args.environment

# directories
home_path = os.path.dirname(os.path.realpath(__file__)) + "/../.."
task_path = os.path.dirname(os.path.realpath(__file__))
weight_path = args.weight.rsplit('/', 1)[0]

# config: take the setup of the existing cfg-file. Optional: define a separate tester cfg
cfg_file = []
for file in os.listdir(weight_path):
    if file.endswith(".yaml"):
        cfg_file.append(file)

if len(cfg_file) > 1:
    raise Exception("Too many cfg-files in the directory, not sure which one to choose")

cfg = YAML().load(open(weight_path + "/" + cfg_file[0], 'r'))

# specify maximum time step
cfg['environment']['max_time'] = 46.0

# create environment from the configuration file
if env_mode == 'target_sequence':
    from raisimGymTorch.env.bin import testenv_target_sequence
    env = VecEnv(testenv_target_sequence.RaisimGymEnv(home_path + "/../rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
                 cfg['environment'], normalize_ob=False)
elif env_mode == 'random_state':
    from raisimGymTorch.env.bin import testenv_random_state
    env = VecEnv(testenv_random_state.RaisimGymEnv(home_path + "/../rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
                 cfg['environment'], normalize_ob=False)
elif env_mode == 'side_hit':
    from raisimGymTorch.env.bin import testenv_side_hit
    env = VecEnv(testenv_side_hit.RaisimGymEnv(home_path + "/../rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
                 cfg['environment'], normalize_ob=False)
else:
    raise Exception("Make sure to select a correct environment. Take a look at the tester-script.")

# shortcuts
ob_dim_expert = env.num_obs
ob_dim_learner = ob_dim_expert - 4
act_dim = env.num_acts
obs = np.zeros((env.num_envs, ob_dim_learner), dtype="float32")
target = np.zeros(shape=(1, 3), dtype="float32")

# to define custom targets outside the Environment.hpp file, uncomment this part. Targets will not be visualized.
#target_list = []
#target_list.append(np.array([10, 10, -2.5], dtype="float32"))
#target_list.append(np.array([0, 20, -5], dtype="float32"))
#target_list.append(np.array([-10, 10, -7.5], dtype="float32"))
#target_list.append(np.array([0, 0, -10], dtype="float32"))
#target_list.append(np.array([10, 10, -12.5], dtype="float32"))

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

    # start recording
    #env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "test"+'.mp4')
    #time.sleep(2)

    # steps
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build and load neural network with deterministic actions
    if cfg['architecture']['shared_nets']:
        loaded_graph = module.sharedBaseNetMLP(cfg['architecture']['base_net'], cfg['architecture']['policy_net'], cfg['architecture']['value_net'],
                                               eval(cfg['architecture']['activation_fn']), ob_dim_learner, [act_dim,1])
        loaded_graph.load_state_dict(torch.load(args.weight.rsplit('/', 1)[0]+"/full_"+str(iteration_number)+'.pt')['actor_architecture_state_dict'])
    else:
        loaded_graph = module.MLP(cfg['architecture']['policy_net'], eval(cfg['architecture']['activation_fn']), ob_dim_learner, act_dim)
        loaded_graph.load_state_dict(torch.load(args.weight.rsplit('/', 1)[0]+"/full_"+str(iteration_number)+'.pt')['actor_architecture_state_dict'])

    # temps
    reward_sum= 0
    done_sum = 0
    done_vec = np.zeros(shape=(int(n_steps * 1.5), cfg["environment"]["num_envs"], 1), dtype="bool")
    current_index = []

    # logging for failed envs
    full_obs = env.observe()
    target = full_obs[0][0:3].copy()
    last_obs = full_obs.copy()
    trajectory = []
    average_task_duration = 0
    task_finished = np.zeros(env.num_envs, dtype=bool)
    fin_count = 0
    path_length = 0
    iterator = 0

    """ Actual testing loop"""
    for step in range(int(n_steps)):
        frame_start = time.time()
        [obs, expert_actions] = np.hsplit(full_obs, [18])
        eval_obs = obs.copy()

        # putting sensor noise on the received observation here instead, to have correct eval_obs
        obs = obs * (1 + np.random.normal(0,0.05))
        obs = helper.normalize_observation(obs)

        # to use custom targets outside the Environment.hpp file, uncomment this part
        #obs[0][0:3] -= target_list[iterator]
        #if step == 800 or step == 1600 or step == 2400 or step == 3200:
        #    print(step)
        #    iterator+= 1
        #    print(iterator)

        # save current position every n-th step
        if step % 10 == 0:
            trajectory.append(eval_obs[0][0:3] - target)

        # check if target is reached within specific tolerance
        for i in range(env.num_envs):
            path_length += np.sqrt(np.power(eval_obs[i][0]-last_obs[i][0],2) + np.power(eval_obs[i][1]-last_obs[i][1],2) + np.power(eval_obs[i][2]-last_obs[i][2],2))
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

        last_obs = full_obs.copy()
        full_obs = env.observe()

        # slow down visualization
        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
        if wait_time > 0:
            time.sleep(wait_time)

        if sum(dones) >= 1 or step == (int(n_steps) - 1):
            num_failed_envs, index = helper.identify_failed_envs(done_vec)
            new_index = [env for env in index if env not in current_index]
            current_index = index
            if len(new_index) > 0:
                print('----------------------------------------------------')
                print('{:<40} {:>6}'.format("total reward: ", '{:0.10f}'.format(reward_sum)))
                print('{:<40} {:>6}'.format("dones: ", '{:0.10f}'.format(done_sum)))
                print('{:<40} {:>6}'.format("failed environments: ", '{:0.6f}'.format(num_failed_envs)))
                print('----------------------------------------------------\n')
            new_index.clear()

    env.stop_video_recording()
    env.turn_off_visualization()

    """ Statistics """

    print("Terminal training step is reached")
    print('{:<40} {:>6}'.format("total reward: ", '{:0.10f}'.format(reward_sum)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.10f}'.format(done_sum)))

    # print average task duration and path length
    if fin_count != 0:
        print('{:<40} {:>6}'.format("average task duration: ", '{:0.10f}'.format(average_task_duration*cfg['environment']['control_dt'] / fin_count)))
        print('{:<40} {:>6}'.format("finished tasks: ", '{:0.10f}'.format(fin_count)))
    print('{:<40} {:>6}'.format("average path_length: ", '{:0.10f}'.format(path_length/ env.num_envs)))

    # export csv-files
    np.savetxt(os.path.join(task_path + "/outputs", 'agent_trajectory_' + str(env_mode) + '.csv'), trajectory, delimiter=",")
