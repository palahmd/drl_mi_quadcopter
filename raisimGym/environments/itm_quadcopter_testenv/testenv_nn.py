from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import itm_quadcopter_testenv
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


# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()

# directories
home_path = os.path.dirname(os.path.realpath(__file__)) + "/../.."
task_path = os.path.dirname(os.path.realpath(__file__))
weight_path = args.weight.rsplit('/', 1)[0]

# config
cfg = YAML().load(open(task_path + "/test_cfg.yaml", 'r'))

# create environment from the configuration file

env = VecEnv(itm_quadcopter_testenv.RaisimGymEnv(home_path + "/../rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], normalize_ob=False)
# shortcuts
ob_dim_expert = env.num_obs
ob_dim_learner = ob_dim_expert - 4
act_dim = env.num_acts
obs = np.zeros((env.num_envs, ob_dim_learner), dtype="float32")
target = np.zeros(shape=(1, 3), dtype="float32")

target_list = []
target_list.append(np.array([10, 10, -2.5], dtype="float32"))
target_list.append(np.array([0, 20, -5], dtype="float32"))
target_list.append(np.array([-10, 10, -7.5], dtype="float32"))
target_list.append(np.array([0, 0, -10], dtype="float32"))
target_list.append(np.array([10, 10, -12.5], dtype="float32"))

trajectory = []

weight_path = args.weight
weight_dir = weight_path.rsplit('/', 1)[0] + '/'

helper = helper(env=env, num_obs=ob_dim_learner,
                normalize_ob=cfg['helper']['normalize_ob'],
                update_mean=cfg['helper']['update_mean'],
                clip_action=cfg['helper']['clip_action'],
                scale_action=cfg['helper']['scale_action'])

iteration_number = int(weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0])

helper.load_scaling(weight_path, iteration_number)


if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))
    start = time.time()
    time.sleep(3)
    env.turn_on_visualization()
    env.reset()
    reward_sum= 0
    done_sum = 0
    average_dones = 0.
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1
    start_step_id = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Visualizing and evaluating the policy: ", weight_path)
    #if cfg['architecture']['shared_nets']:
    #   loaded_graph = module.sharedBaseNetMLP(cfg['architecture']['base_net'], cfg['architecture']['policy_net'], cfg['architecture']['value_net'],
    #                                           eval(cfg['architecture']['activation_fn']), ob_dim_learner, [act_dim,1])
    #    loaded_graph.load_state_dict(torch.load(weight_path.rsplit('/', 1)[0]+"/full_"+str(iteration_number)+'.pt')['actor_architecture_state_dict'])
    #else:
    loaded_graph = module.MLP(cfg['architecture']['policy_net'], eval(cfg['architecture']['activation_fn']), ob_dim_learner, act_dim)
    loaded_graph.load_state_dict(torch.load(weight_path.rsplit('/', 1)[0]+"/full_"+str(iteration_number)+'.pt')['actor_architecture_state_dict'])
    MLP=module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0)


    helper.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()
    for i in range(10):
        env.reset()
    #env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "test"+'.mp4')
    #time.sleep(2)

    # temps
    done_vec = np.zeros(shape=(int(n_steps * 1.5), cfg["environment"]["num_envs"], 1), dtype="bool")
    current_index = []
    full_obs = env.observe()
    target = full_obs[0][0:3].copy()
    last_obs = full_obs.copy()

    all_times = 0
    finished = np.zeros(env.num_envs, dtype=bool)
    count = 0
    path_length = 0
    pl_vec = []
    iterator = 0

    for step in range(int(n_steps)):
        frame_start = time.time()
        [obs, expert_actions] = np.hsplit(full_obs, [18])
        eval_obs = obs.copy()
        obs[0][0:3] -= target_list[iterator]

        if step == 800 or step == 1600 or step == 2400 or step == 3200:
            print(step)
            iterator+= 1
            print(iterator)

        if step % 2 == 0:
            trajectory.append(eval_obs[0][0:3])


        #obs = obs * (1 + np.random.normal(0,0.05))

        obs = helper.normalize_observation(obs)


        for i in range(env.num_envs):
            path_length += np.sqrt(np.power(eval_obs[i][0]-last_obs[i][0],2) + np.power(eval_obs[i][1]-last_obs[i][1],2) + np.power(eval_obs[i][2]-last_obs[i][2],2))      
            if np.sqrt(np.power(eval_obs[i][0],2) + np.power(eval_obs[i][1],2) + np.power(eval_obs[i][2],2)) < 0.2:
                if finished[i] == False and step > 200:
                    pl_vec.append(step * cfg["environment"]["control_dt"])
                    all_times += step
                    finished[i] = True
                    count += 1

        #if cfg['architecture']['shared_nets']:
        #    action = loaded_graph.actor_net(torch.from_numpy(obs))
        #else:
        action = loaded_graph.architecture(torch.from_numpy(obs))
        #action, _ = MLP.sample(action)
        action = helper.limit_action(action)


        reward, dones = env.step(action)
        reward_sum+= sum(reward)
        done_sum += sum(dones)
        done_vec[step] = dones.reshape(env.num_envs, 1).copy()
        
        last_obs = full_obs.copy()
        full_obs = env.observe()

        if step == (799 or 1599 or 2399):
            target = full_obs[0][0:3].copy()


        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
        #if wait_time > 0:
        #    time.sleep(wait_time)
        #time.sleep(cfg['environment']['control_dt'])

        if step == n_steps:
            print("Terminal training step is reached")


        if sum(dones) >= 1 or step == (int(n_steps) - 1):
            num_failed_envs, index = helper.identify_failed_envs(done_vec)
            new_index = [env for env in index if env not in current_index]
            current_index = index
            if len(new_index) > 0:
                print('----------------------------------------------------')
                print('{:<40} {:>6}'.format("total reward: ", '{:0.10f}'.format(reward_sum)))
                print('{:<40} {:>6}'.format("dones: ", '{:0.10f}'.format(done_sum)))
                print('{:<40} {:>6}'.format("failed environments: ", '{:0.6f}'.format(num_failed_envs)))
            new_index.clear()

    print('{:<40} {:>6}'.format("total reward: ", '{:0.10f}'.format(reward_sum)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.10f}'.format(done_sum)))

    np.savetxt(os.path.join(task_path, 'nn_trajectory_single_target_neg_550.csv'), trajectory, delimiter=",")
    
    if count != 0:
            print(all_times * cfg["environment"]["control_dt"]/count)
            print(count)
    print(path_length/cfg["environment"]["num_envs"])

    env.stop_video_recording()
    env.turn_off_visualization()
    env.reset()
    print("Finished at the maximum visualization steps")
