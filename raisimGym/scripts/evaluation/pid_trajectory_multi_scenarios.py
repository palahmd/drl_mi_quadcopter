from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.env_helper import helper
from raisimGymTorch.algo.pid_controller.pid_controller import PID
import os
import math
import time
import datetime
import argparse
import numpy as np

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--environment', help='set environment', type=str, default='target_tracking')
args = parser.parse_args()
env_mode = args.environment

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../.."
start_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# config
cfg = YAML().load(open(task_path + "/" + "pid_" + "cfg.yaml", 'r'))

# specify maximum time step
cfg['environment']['max_time'] = 15.0

# create environment from the configuration file
if env_mode == 'target_tracking':
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
ob_dim = env.num_obs
act_dim = env.num_acts
actions = np.zeros(shape=(env.num_envs, act_dim), dtype="float32")
targets = np.zeros(shape=(env.num_envs, ob_dim), dtype="float32")
init_state = np.zeros(shape=(cfg['environment']['num_envs'], ob_dim), dtype="float32")
for i in range(len(init_state)):
    init_state[i][2] = 0.135
    init_state[i][3] = 1
    init_state[i][7] = 1
    init_state[i][11] = 1
    init_state[i][18] = 1

# to define custom targets outside the Environment.hpp file, uncomment this part. Targets will not be visualized.
#target_list = []
#target_list.append(np.array([10, 10, -2.5], dtype="float32"))
#target_list.append(np.array([0, 20, -5], dtype="float32"))
#target_list.append(np.array([-10, 10, -7.5], dtype="float32"))
#target_list.append(np.array([0, 0, -10], dtype="float32"))
#target_list.append(np.array([10, 10, -12.5], dtype="float32"))

# environment helper for specific operations
helper = helper(env=env, num_obs=ob_dim,
                normalize_ob=cfg['helper']['normalize_ob'],
                update_mean=cfg['helper']['update_mean'],
                clip_action=cfg['helper']['clip_action'],
                scale_action=cfg['helper']['scale_action'],
                )

# task and PID parameters
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs
pid = PID(1.5, 50, 4.1, ob_dim, act_dim, cfg['environment']['control_dt'], 1.727)

env.turn_on_visualization()
env.start_video_recording("stage_2_pid_4s.mp4")
pos_controller_loopCount = 0
time.sleep(2.5)

# set target
obs = env.observe()
target = obs[0][0:3].copy()
last_obs = obs.copy()
#targets = init_state - obs.copy()

# temps; dones = task failures
reward_sum = 0
done_sum = 0
average_dones = 0.

done_vec = np.zeros(shape=(int(n_steps * 1.5), cfg["environment"]["num_envs"], 1), dtype="bool")
average_task_duration = 10e-5 # to prevent div by zero
task_finished = np.zeros(env.num_envs, dtype=bool)
fin_count = 0
path_length = 0
iterator = 0
trajectory = []

for step in range(int(n_steps)):
    frame_start = time.time()
    #obs += targets
    eval_obs = obs.copy()
    #targets[0][0:3] = target_list[iterator]
    obs = obs * (1 + np.random.normal(0,0.05))

    #if step == 800 or step == 1600 or step == 2400 or step == 3200:
    #    print(step)
    #    iterator+= 1
    #    print(iterator)
        #target = targets[iterator]

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

    for i in range(0, env.num_envs):
        expert_obs_env_i = obs[i, :]
        actions[i, :] = pid.control(obs=expert_obs_env_i.reshape((ob_dim, 1)),
                                              target=targets[i][0:12].reshape((12, 1)), pos_controller_loopCount=pos_controller_loopCount)

    reward, dones = env.step(actions)
    reward_sum += sum(reward)
    done_sum += sum(dones)
    done_vec[step] = dones.reshape(env.num_envs, 1).copy()

    last_obs = eval_obs.copy()
    obs = env.observe()

    # frequency of outer PID position controller can be set here (also rule-based)
    if pos_controller_loopCount == 8:
        pos_controller_loopCount = 7
        #if step >= n_steps/4:
        #   pos_controller_loopCount = 3
    pos_controller_loopCount += 1

    frame_end = time.time()

    wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
    if wait_time > 0.:
        time.sleep(wait_time)


env.turn_off_visualization()
env.stop_video_recording()

""" Statistics """

num_failed_envs, index = helper.identify_failed_envs(done_vec)

print('----------------------------------------------------')
print('{:<40} {:>6}'.format("total dones: ", '{:0.6f}'.format(done_sum)))
print('{:<40} {:>6}'.format("failed environments: ", '{:0.6f}'.format(num_failed_envs)))
print('{:<40} {:>6}'.format("total reward: ", '{:0.6f}'.format(reward_sum)))
print('----------------------------------------------------\n')

# print average task duration and path length
if fin_count != 0:
    print('{:<40} {:>6}'.format("average task duration: ", '{:0.10f}'.format(average_task_duration*cfg['environment']['control_dt'] / fin_count)))
    print('{:<40} {:>6}'.format("finished tasks: ", '{:0.10f}'.format(fin_count)))
print('{:<40} {:>6}'.format("average path_length: ", '{:0.10f}'.format(path_length/ env.num_envs)))

# export csv-files
np.savetxt(os.path.join(task_path + "/outputs", 'pid_trajectory_' + str(env_mode) + '.csv'), trajectory, delimiter=",")

