import torch
import numpy as np

def normalize_action(actions):
    for i in range(0, len(actions)):
        min = torch.min(actions[i][:])
        max = torch.max(actions[i][:])

        if torch.abs(min) > 1 or torch.abs(max) > 1:
            if torch.abs(min) < torch.abs(max):
                actions[i][:] /= torch.abs(max)
            else:
                actions[i][:] /= torch.abs(min)

    return actions

def normalize_observation(env, observation, normalize_ob=True, update_mean=True):
    if normalize_ob == True:
        if update_mean:
            env.obs_rms.update(observation)

        observation_norm = np.clip((observation - env.obs_rms.mean) / np.sqrt(env.obs_rms.var + 1e-8), -env.clip_obs,
                    env.clip_obs)

        return observation_norm

    else:
        return observation