from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import itm_quadcopter_dagger
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
cfg = YAML().load(open(weight_path + "/dagger_cfg.yaml", 'r'))

# create environment from the configuration file
#cfg['environment']['num_envs'] = 1

env = VecEnv(itm_quadcopter_dagger.RaisimGymEnv(home_path + "/../rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], normalize_ob=False)
# shortcuts
ob_dim_expert = env.num_obs
ob_dim_learner = ob_dim_expert - 4
act_dim = env.num_acts
obs = np.zeros((env.num_envs, ob_dim_learner), dtype="float32")
targets = np.zeros((env.num_envs, 3), dtype="float32")

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
    if cfg['architecture']['shared_nets']:
        loaded_graph = module.sharedBaseNetMLP(cfg['architecture']['base_net'], cfg['architecture']['policy_net'], cfg['architecture']['value_net'],
                                               eval(cfg['architecture']['activation_fn']), ob_dim_learner, [act_dim,1])
        loaded_graph.load_state_dict(torch.load(weight_path.rsplit('/', 1)[0]+"/full_"+str(iteration_number)+'.pt')['actor_architecture_state_dict'])
    else:
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
    for i in range(env.num_envs):
        targets[i] = full_obs[i][0:3].copy()
    
    for step in range(int(n_steps*1.5)):
        frame_start = time.time()
        [obs, expert_actions] = np.hsplit(full_obs, [18])
        obs = helper.normalize_observation(obs)

        if cfg['architecture']['shared_nets']:
            action = loaded_graph.actor_net(torch.from_numpy(obs))
        else:
            action = loaded_graph.architecture(torch.from_numpy(obs))
            #action, _ = MLP.sample(action)
        action = helper.limit_action(action)


        reward, dones = env.step(action)
        reward_sum+= sum(reward)
        done_sum += sum(dones)
        done_vec[step] = dones.reshape(env.num_envs, 1).copy()

        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
        if wait_time > 0:
            time.sleep(wait_time)
        #time.sleep(cfg['environment']['control_dt'])

        full_obs = env.observe()

        if sum(dones) >= 1 or step == (int(n_steps*1.5) - 1):
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

    print('{:<40} {:>6}'.format("total reward: ", '{:0.10f}'.format(reward_sum)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.10f}'.format(done_sum)))

    env.stop_video_recording()
    env.turn_off_visualization()
    env.reset()
    print("Finished at the maximum visualization steps")
