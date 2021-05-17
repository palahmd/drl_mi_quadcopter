import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import random

class RolloutStorage:
    def __init__(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape, device):
        self.device = device

        # Core
        self.actor_obs = torch.zeros(num_transitions_per_env, num_envs, *actor_obs_shape).to(self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1).byte().to(self.device)

        # For DAgger and BC
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)
        self.expert_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape).to(self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.device = device
        self.num_acts = actions_shape[0]
        self.num_obs = critic_obs_shape[0]

        self.step = 0
        self.init_state = torch.zeros(critic_obs_shape[0]).to(self.device)
        self.init_state[2] = 0.135
        self.init_state[3] = 1
        self.init_state[7] = 1
        self.init_state[11] = 1
        self.init_action = torch.zeros(actions_shape[0]).to(self.device)

    def add_transitions(self, actor_obs, expert_actions, rewards, dones, values):
        #if self.step >= self.num_transitions_per_env:
         #   raise AssertionError("Rollout buffer overflow")
        self.actor_obs[self.step].copy_(torch.from_numpy(actor_obs).to(self.device))
        self.expert_actions[self.step].copy_(expert_actions.to(self.device))
        self.rewards[self.step].copy_(torch.from_numpy(rewards).view(-1, 1).to(self.device))
        self.dones[self.step].copy_(torch.from_numpy(dones).view(-1, 1).to(self.device))
        self.values[self.step].copy_(values.to(self.device))
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]

            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        #self.advantages = self.returns - self.values
        #self.advantages = (self.advantages - self.advantages.mean()) / #(self.advantages.std() + 1e-8)

    def mini_batch_generator_shuffle(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for indices in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True):
            actor_obs_batch = self.actor_obs.view(-1, *self.actor_obs.size()[2:])[indices]
            expert_actions_batch = self.expert_actions.view(-1, self.expert_actions.size(-1))[indices]
            values_batch = self.values.view(-1, 1)[indices]
            returns_batch = self.returns.view(-1, 1)[indices]
            dones_batch = self.dones.view(-1, 1)[indices]
            yield actor_obs_batch, expert_actions_batch, \
                  values_batch, returns_batch, dones_batch

    def mini_batch_generator_inorder(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for batch_id in range(num_mini_batches):
            yield self.actor_obs.view(-1, *self.actor_obs.size()[2:])[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.expert_actions.view(-1, self.expert_actions.size(-1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.values.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.returns.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.dones.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size]


    def reset_failed_episodes(self):
        failed_envs = torch.where(self.dones == 1)
        index = failed_envs[1].tolist()

        for env in range(len(index)-1):
            for transition in range(len(self.dones)):
                self.actor_obs[transition][index[env]] = self.init_state
                self.expert_actions[transition][index[env]] = self.init_action
                self.rewards[transition][index[env]] = 0


    def filter_failed_envs(self, filter_envs = True):
        failed_envs = torch.where(self.dones == 1)
        failed_envs_index = list(dict.fromkeys(failed_envs[1].tolist()))
        num_failed_envs = len(failed_envs_index)

        if num_failed_envs > 0 and filter_envs:
            compl_envs = torch.where(self.dones == 0)
            compl_envs_index = [env for env in list(dict.fromkeys(compl_envs[1].tolist())) if env not in failed_envs_index]
            for env in range(num_failed_envs):
                rand_num = random.randint(0, len(compl_envs_index)-1)
                for transition in range(self.num_transitions_per_env):
                    self.actor_obs[transition][failed_envs_index[env]] = self.actor_obs[transition][compl_envs_index[rand_num]]
                    self.expert_actions[transition][failed_envs_index[env]] = self.expert_actions[transition][compl_envs_index[rand_num]]
                    self.values[transition][failed_envs_index[env]] = self.values[transition][compl_envs_index[rand_num]]
                    self.returns[transition][failed_envs_index[env]] =self.returns[transition][compl_envs_index[rand_num]]
                    self.dones[transition][failed_envs_index[env]] = self.dones[transition][compl_envs_index[rand_num]]

        return num_failed_envs
