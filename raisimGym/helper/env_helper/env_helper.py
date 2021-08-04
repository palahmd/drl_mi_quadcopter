import torch
import numpy as np
import os
import ntpath
from shutil import copyfile
from raisimGymTorch.env.RaisimGymVecEnv import RunningMeanStd

class helper:

    def __init__(self, env, num_obs, normalize_ob=True, update_mean=True, scale_action=True, clip_action=False,
                 scale_obs_rms=False):
        self.num_envs = env.num_envs
        self.num_obs = num_obs
        self.obs_rms = RunningMeanStd(shape=[self.num_envs, self.num_obs])
        self.clip_obs = env.clip_obs
        self.normalize_ob = normalize_ob
        self.update_mean = update_mean
        self.scale_action = scale_action
        self.clip_action = clip_action
        self.scale_obs_rms = scale_obs_rms

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

    """ works as an environment wrapper, uses methods of env to normalize the observation and update the RMS.
        when to use: If the observation vector of env.observe has more entries than the actual neural network input. 
        The respective method of the env object normalizes also the additional entries of the observation vector 
        passed from the environment (defined in Environment.hpp)"""
    def normalize_observation(self, observation):
        if self.normalize_ob == True:
            if self.update_mean:
                # update observation scaling based on the parallel algorithm
                self.obs_rms.update(observation)

            """ if scale_obs_rms = True:
                observation RMS will be scaled which has the effect of projecting the target point further away or closer
                to the agent with a scaling factor of (0.2 + norm(pos)/10), since the agent is trained on target
                points which are 10 m away."""
            obs_rms_var = self.obs_rms.var.copy()
            if self.scale_obs_rms:
                if len(observation) > 1:
                    for i in range(len(obs_rms_var)):
                        obs_rms_var[i][0:3] = self.obs_rms.var[i][0:3] * (0.2+(np.linalg.norm(observation[i][0:3])/10)*0.8)

                    observation_norm = np.clip((observation - self.obs_rms.mean) / np.sqrt(obs_rms_var + 1e-8), - self.clip_obs,
                                               self.clip_obs)
                else:
                    obs_rms_var[0:3] = self.obs_rms.var[0:3] * (0.2+(np.linalg.norm(observation[0:3])/10)*0.8)
                    observation_norm = np.clip((observation - self.obs_rms.mean[0]) / np.sqrt(obs_rms_var[0] + 1e-8), - self.clip_obs,
                                               self.clip_obs)
            else:
                observation_norm = np.clip((observation - self.obs_rms.mean) / np.sqrt(obs_rms_var + 1e-8), - self.clip_obs,
                                       self.clip_obs)

            return observation_norm

        else:
            return observation

    # save and load observation scaling
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

    # load the neural network model parameters
    def load_param(self, weight_path, actor, critic, learner, data_dir, file_name, save_items=True):
        if weight_path == "":
            raise Exception("\nCan't find the pre-trained weight, please provide a pre-trained weight with --weight switch\n")
        print("\nRetraining from the checkpoint:", weight_path+"\n")

        iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
        weight_dir = weight_path.rsplit('/', 1)[0] + '/'

        mean_csv_path = weight_dir + 'mean' + iteration_number + '.csv'
        var_csv_path = weight_dir + 'var' + iteration_number + '.csv'

        items_to_save = [weight_path, mean_csv_path, var_csv_path, weight_dir +file_name + "cfg.yaml", weight_dir +
                             "Environment.hpp"]

        if items_to_save is not None and save_items:
            pretrained_data_dir = data_dir + '/pretrained_' + weight_path.rsplit('/', 1)[0].rsplit('/', 1)[1]
            os.makedirs(pretrained_data_dir)
            for item_to_save in items_to_save:
                copyfile(item_to_save, pretrained_data_dir+'/'+item_to_save.rsplit('/', 1)[1])

        # load actor and critic parameters from full checkpoint
        checkpoint = torch.load(weight_path)
        actor.architecture.load_state_dict(checkpoint['actor_architecture_state_dict'])
        actor.distribution.load_state_dict(checkpoint['actor_distribution_state_dict'])
        critic.architecture.load_state_dict(checkpoint['critic_architecture_state_dict'])
        #learner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #learner.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        """ Only useful in combination with DAgger: A method to prevent learning from failed environments and restoring 
        from the last checkpoint"""
    def restart_from_last_checkpoint(self, env, saver, actor, critic, learner, update_num):
        # Reset update number
        update_modulo = update_num % 10

        # Reset learner params
        learner.storage.clear()
        learner.beta += update_modulo * learner.beta_scheduler
        learner.scheduler.step(epoch=(update_num-update_modulo)*learner.num_learning_epochs*learner.num_mini_batches)
        learner.beta += learner.beta_scheduler

        # Set new environment target
        for i in range(10 - update_modulo - 1):
            env.reset()

        # Restore weights from last checkpoint
        weight_path = saver.data_dir + "/full_" + str(update_num - update_modulo) + '.pt'
        self.load_param(weight_path, actor, critic, learner, saver.data_dir, 'dagger', False)

        return update_modulo + 1

    def identify_failed_envs(self, dones) -> object:
        failed_envs = np.where(dones == 1)
        index = list(dict.fromkeys(failed_envs[1].tolist()))

        return  len(index), index
