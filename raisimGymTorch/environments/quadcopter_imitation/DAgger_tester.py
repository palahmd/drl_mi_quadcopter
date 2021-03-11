from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import quadcopter_imitation
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import raisimGymTorch.algo.shared_modules.actor_critic as module
from raisimGymTorch.algo.imitation_learning.DAgger import DAgger
from raisimGymTorch.helper.env_helper.env_helper import scale_action, normalize_observation
import os
import math
import time
import torch
import argparse
import numpy as np
import torch.nn as nn


# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()

# directories
home_path = os.path.dirname(os.path.realpath(__file__)) + "/../../.."
task_path = os.path.dirname(os.path.realpath(__file__))

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
cfg['environment']['num_envs'] = 1

env = VecEnv(quadcopter_imitation.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'],
                                                                            Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim_expert = env.num_obs
ob_dim_learner = ob_dim_expert - 4
act_dim = env.num_acts
target_point = np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype="float32")

weight_path = args.weight
#iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
iteration_number = 320
weight_dir = weight_path.rsplit('/', 1)[0] + '/'

if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))
    start = time.time()
    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1
    start_step_id = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO: Test if sampling over distribution works better
    print("Visualizing and evaluating the policy: ", weight_path)
    loaded_graph = module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim_learner, act_dim)
    loaded_graph.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()

    # max_steps = 1000000
    max_steps = 1000 ## 10 secs

    for step in range(max_steps):
        time.sleep(0.01)
        expert_ob = env.observe(update_mean=False) # expert_obs dimension = 22. The last four are quaternions
        expert_ob_clipped = expert_ob.copy()
        expert_ob_clipped.resize((1, 18), refcheck=False)
        learner_ob = target_point - expert_ob_clipped
        action_ll = loaded_graph.architecture(torch.from_numpy(learner_ob).cpu())
        normalize_action(action_ll)
        reward_ll, dones = env.step(action_ll.cpu().detach().numpy())
        reward_ll_sum = reward_ll_sum + reward_ll[0]
        if dones or step == max_steps - 1:
            print('----------------------------------------------------')
            print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(reward_ll_sum / (step + 1 - start_step_id))))
            print('{:<40} {:>6}'.format("time elapsed [sec]: ", '{:6.4f}'.format((step + 1 - start_step_id) * 0.01)))
            print('----------------------------------------------------\n')
            start_step_id = step + 1
            reward_ll_sum = 0.0

    env.turn_off_visualization()
    env.reset()
    print("Finished at the maximum visualization steps")
