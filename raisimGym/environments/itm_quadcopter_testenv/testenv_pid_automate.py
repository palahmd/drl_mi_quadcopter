from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import itm_quadcopter_testenv
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.env_helper import helper
import os
import math
import time
from raisimGymTorch.algo.pid_controller.pid_controller import PID
import numpy as np
import torch

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
file_name = ""
if len(os.path.basename(__file__).split("_", 1)) != 1:
    for i in range(len(os.path.basename(__file__).split("_", 1)) - 1):
        file_name += os.path.basename(__file__).split("_", 1)[0] + "_"
home_path = os.path.dirname(os.path.realpath(__file__)) + "/../.."
task_path = os.path.dirname(os.path.realpath(__file__))

# config
cfg = YAML().load(open(task_path + "/" + "test_cfg.yaml", 'r'))

# create environment from the configuration file
env = VecEnv(itm_quadcopter_testenv.RaisimGymEnv(home_path + "/../rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], normalize_ob=False)

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

trajectory = np.zeros(shape=(math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt']), ob_dim))


helper = helper(env=env, num_obs=ob_dim,
                normalize_ob=cfg['helper']['normalize_ob'],
                update_mean=cfg['helper']['update_mean'],
                clip_action=cfg['helper']['clip_action'],
                scale_action=cfg['helper']['scale_action'])

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs
pid = PID(1.5, 50, 4.1, ob_dim, act_dim, cfg['environment']['control_dt'], 1.727)
#pid = PID(2.5, 200, 8, ob_dim, act_dim, cfg['environment']['control_dt'], 1.727)

pl_vec = []
dist = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for d in range(len(dist)):
    pl_vec.append([])
    for i in range(10):
        env.reset()
    env.turn_on_visualization()
    loopCount = 0
    time.sleep(0.5)
    obs = env.observe()
    last_obs = obs.copy()
    #targets = init_state - obs.copy()

    reward_sum = 0
    done_sum = 0
    average_dones = 0.

    done_vec = np.zeros(shape=(int(n_steps * 1.5), cfg["environment"]["num_envs"], 1), dtype="bool")
    all_times = 0
    finished = np.zeros(env.num_envs, dtype=bool)
    count = 0
    path_length = 0
    for step in range(int(n_steps)):
        frame_start = time.time()
        #obs += targets
        trajectory[step] = obs[0]
        eval_obs = obs.copy()
        obs = obs * (1 + np.random.normal(0,0.05))
        
        for i in range(0, env.num_envs):
            expert_obs_env_i = obs[i, :]
            actions[i, :] = pid.control(obs=expert_obs_env_i.reshape((ob_dim, 1)),
                                                  target=targets[i][0:12].reshape((12, 1)), loopCount=loopCount)

        reward, dones = env.step(actions)
        reward_sum += sum(reward)
        done_sum += sum(dones)
        done_vec[step] = dones.reshape(env.num_envs, 1).copy()


        for i in range(env.num_envs):
            path_length += np.sqrt(np.power(eval_obs[i][0]-last_obs[i][0],2) + np.power(eval_obs[i][1]-last_obs[i][1],2) + np.power(eval_obs[i][2]-last_obs[i][2],2))      
            if np.sqrt(np.power(eval_obs[i][0],2) + np.power(eval_obs[i][1],2) + np.power(eval_obs[i][2],2)) < dist[d]:
                if finished[i] == False and step > 200:
                    pl_vec[d].append(step * cfg["environment"]["control_dt"])
                    all_times += step
                    finished[i] = True
                    count += 1
        
        last_obs = eval_obs.copy()                    
        obs = env.observe()

        # frequency of outter PID acceleration controller
        if loopCount == 8:
            loopCount = 7
            #if step >= n_steps/4:
             #   loopCount = 3
        loopCount += 1

        frame_end = time.time()

        wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
        #if wait_time > 0.:
        #    time.sleep(wait_time)

    num_failed_envs, index = helper.identify_failed_envs(done_vec)

    print('----------------------------------------------------')
    print('{:<40} {:>6}'.format("total dones: ", '{:0.6f}'.format(done_sum)))
    print('{:<40} {:>6}'.format("failed environments: ", '{:0.6f}'.format(num_failed_envs)))
    print('{:<40} {:>6}'.format("total reward: ", '{:0.6f}'.format(reward_sum)))
    print('----------------------------------------------------\n')

    if count != 0:
        print(all_times * cfg["environment"]["control_dt"]/count)
        print(count)
    print(path_length/cfg["environment"]["num_envs"])


    start_step_id = step + 1
    reward_ll_sum = 0.0

np.savetxt(os.path.join(task_path, 'pid_duration_0.1-1.0.csv'), pl_vec, delimiter=",")
env.turn_off_visualization()
