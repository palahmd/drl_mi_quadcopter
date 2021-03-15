import torch
import numpy as np
from raisimGymTorch.env.RaisimGymVecEnv import RunningMeanStd

class helper:

    def __init__(self, env, num_obs, normalize_ob=True, update_mean=True, scale_action=True, clip_action=False):
        self.num_envs = env.num_envs
        self.num_obs = num_obs
        self.obs_rms = RunningMeanStd(shape=[self.num_envs, self.num_obs])
        self.clip_obs = env.clip_obs
        self.normalize_ob = normalize_ob
        self.update_mean = update_mean
        self.scale_action = scale_action
        self.clip_action = clip_action


    # action scaling
    def limit_action(self, actions):
        if self.clip_action:
            return np.clip(actions.cpu().detach().numpy(), -1, 1)
        elif self.scale_action:
            for i in range(0, len(actions)):
                min = torch.min(actions[i][:])
                max = torch.max(actions[i][:])

                if torch.abs(min) > 1 or torch.abs(max) > 1:
                    if torch.abs(min) < torch.abs(max):
                        actions[i][:] /= torch.abs(max)
                    else:
                        actions[i][:] /= torch.abs(min)

            return actions.cpu().detach().numpy()
        else:
            return actions.cpu().detach().numpy()


    # works as an environment wrapper, uses methods of env to normalize the observation and update the RMS.
    # when to use: if a target point is defined in the runner file and needs to be calculated into the observation
    def normalize_observation(self, observation):
        if self.normalize_ob == True:
            if self.update_mean:
                self.obs_rms.update(observation)

            observation_norm = np.clip((observation - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), - self.clip_obs,
                        self.clip_obs)

            return observation_norm

        else:
            return observation

    def save_scaling(self, dir_name, iteration):
        mean_file_name = dir_name + "/mean" + iteration + ".csv"
        var_file_name = dir_name + "/var" + iteration + ".csv"
        np.savetxt(mean_file_name, self.obs_rms.mean)
        np.savetxt(var_file_name, self.obs_rms.var)

    def load_scaling(self, weight_dir, iteration, count=1e5):
        dir_name = weight_dir.rsplit('/', 1)[0]
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.obs_rms.count = count
        self.obs_rms.mean = np.loadtxt(mean_file_name, dtype=np.float32)
        self.obs_rms.var = np.loadtxt(var_file_name, dtype=np.float32)