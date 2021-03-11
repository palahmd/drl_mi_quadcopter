import torch
import numpy as np

# action clipping instead of scaling
def clip_action(actions,  clip_action=False):
    if clip_action:
        return np.clip(actions.cpu().detach().numpy(), -1, 1)
    else:
        return actions.cpu().detach().numpy()


# action scaling
def scale_action(actions):
    for i in range(0, len(actions)):
        min = torch.min(actions[i][:])
        max = torch.max(actions[i][:])

        if torch.abs(min) > 1 or torch.abs(max) > 1:
            if torch.abs(min) < torch.abs(max):
                actions[i][:] /= torch.abs(max)
            else:
                actions[i][:] /= torch.abs(min)

    return actions.cpu().detach().numpy()
    

# works as an environment wrapper, uses methods of env to normalize the observation and update the RMS.
# when to use: if a target point is defined in the runner file and needs to be calculated into the observation
def normalize_observation(env, observation, normalize_ob=True, update_mean=True):
    if normalize_ob == True:
        if update_mean:
            env.obs_rms.update(observation)

        observation_norm = np.clip((observation - env.obs_rms.mean) / np.sqrt(env.obs_rms.var + 1e-8), -env.clip_obs,
                    env.clip_obs)

        return observation_norm

    else:
        return observation
