import torch
import numpy as np
import os
import ntpath
from shutil import copyfile
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

            max_ob = observation[0:3].max()
            if max_ob > self.clip_obs:
                observation[0:3] /= (max_ob/self.clip_obs)
                return observation

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

    def restart_from_last_checkpoint(self, env, saver, actor, critic, learner, update_num):
        """ ONLY USEFUL FOR DAGGER_RUNNER"""
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
