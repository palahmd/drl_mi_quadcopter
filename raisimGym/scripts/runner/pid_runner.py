from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.env_helper import helper
from raisimGymTorch.algo.pid_controller.pid_controller import PID
import os
import math
import time
import datetime
import numpy as np
import argparse

# configuration: run the pid-controller in different environments
# python dagger_runner.py -e stage_1
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--environment', help='set environment', type=str, default='stage_1')
args = parser.parse_args()
env_mode = args.environment

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../.."
start_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# config
cfg = YAML().load(open(task_path + "/" + "pid_" + "cfg.yaml", 'r'))

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
    raise Exception("Make sure to select a correct environment. Take a look at the runner-script")

# shortcuts 
ob_dim = env.num_obs
act_dim = env.num_acts
actions = np.zeros(shape=(env.num_envs, act_dim), dtype="float32")

# optinal: define custom targets outside the Environment.hpp file
targets = np.zeros(shape=(env.num_envs, ob_dim), dtype="float32")
init_state = np.zeros(shape=(cfg['environment']['num_envs'], ob_dim), dtype="float32")
for i in range(len(init_state)):
    init_state[i][2] = 0.135
    init_state[i][3] = 1
    init_state[i][7] = 1
    init_state[i][11] = 1
    init_state[i][18] = 1

# environment helper for specific operations
helper = helper(env=env, num_obs=ob_dim,
                normalize_ob=cfg['helper']['normalize_ob'],
                update_mean=cfg['helper']['update_mean'],
                clip_action=cfg['helper']['clip_action'],
                scale_action=cfg['helper']['scale_action'])

# task and PID parameters
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs
pid = PID(1.5, 50, 4.1, ob_dim, act_dim, cfg['environment']['control_dt'], 1.727)

for update in range(10):
    # to make sure new target is generated (e.g. in the task of stage 1)
    for i in range(10):
        env.reset()
        
    # turn on visualization and wait until it is loaded
    env.turn_on_visualization()
    # env.start_video_recording(start_date + "policy_" + str(update) + '.mp4')
    time.sleep(0.5)
    
    # reset counter for the position controller and get target
    pos_controller_loopCount = 0
    obs = env.observe()

    # temps; dones = task failures
    reward_sum = 0
    done_sum = 0
    average_dones = 0.

    done_vec = np.zeros(shape=(int(n_steps * 1.5), cfg["environment"]["num_envs"], 1), dtype="bool")
    average_task_duration = 10e-5 # to prevent div by zero
    task_finished = np.zeros(env.num_envs, dtype=bool)
    fin_count = 0

    for step in range(int(n_steps)):
        frame_start = time.time()

        # takes too much resources, perhaps can be solved more elegantly without a for-loop and large numpy-arrays
        for i in range(0, env.num_envs):
            expert_obs_env_i = obs[i, :]
            actions[i, :] = pid.control(obs=expert_obs_env_i.reshape((ob_dim, 1)),
                                                  target=targets[i][0:12].reshape((12, 1)), pos_controller_loopCount=pos_controller_loopCount)

        # reward and done logging
        reward, dones = env.step(actions)
        reward_sum += sum(reward)
        done_sum += sum(dones)
        done_vec[step] = dones.reshape(env.num_envs, 1).copy()

        obs = env.observe()

        # calculate avarage durations how long it takes to finish task. Define an area within task is regarded finished
        for i in range(env.num_envs):
            if np.sqrt(np.power(obs[i][0],2) + np.power(obs[i][1],2) + np.power(obs[i][2],2)) < 0.1:
                if task_finished[i] == False and step > 200:
                    average_task_duration += step
                    task_finished[i] = True
                    fin_count += 1

        # frequency of outer PID position controller can be set here (also rule-based)
        if pos_controller_loopCount == 8:
            pos_controller_loopCount = 7
            #if step >= n_steps/4:
             #   pos_controller_loopCount = 3
        pos_controller_loopCount += 1

        # slow down visualization
        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)

    num_failed_envs, index = helper.identify_failed_envs(done_vec)

    print('----------------------------------------------------')
    print('{:<40} {:>6}'.format("total dones: ", '{:0.6f}'.format(done_sum)))
    print('{:<40} {:>6}'.format("failed environments: ", '{:0.6f}'.format(num_failed_envs)))
    print('{:<40} {:>6}'.format("total reward: ", '{:0.6f}'.format(reward_sum)))
    print('{:<40} {:>6}'.format("average task duration: ", '{:0.6f}'.format(average_task_duration*cfg['environment']['control_dt'] / fin_count)))
    print('----------------------------------------------------\n')

    # env.stop_video_recording()

env.turn_off_visualization()
