"""
This file is partly based on the respective file of the original raisimGymTorch repository with modifications.
"""

from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from .ppo_storage import RolloutStorage


class PPO:
    def __init__(self,
                 actor,
                 critic,
                 num_envs,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=0.5,
                 entropy_coef=0.0,
                 learning_rate=5e-4,
                 max_grad_norm=0.5,
                 l2_reg_coef=0.0001,
                 bc_coef=0.005,
                 use_clipped_value_loss=True,
                 log_dir='run',
                 device='cpu',
                 shuffle_batch=True,
                 deterministic_policy=False):

        # env parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # PPO components
        self.actor = actor
        self.critic = critic

        # data storage, containing observations, actions, rewards etc.
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor.obs_shape, critic.obs_shape, actor.action_shape, device)

        if shuffle_batch:
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        else:
            self.batch_sampler = self.storage.mini_batch_generator_inorder

        self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=learning_rate)
        self.device = device

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.deterministic_policy = deterministic_policy
        self.l2_reg_coef = l2_reg_coef
        self.bc_coef = bc_coef

        # Tensorboard logger
        self.log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0

        # temps
        self.actions = None
        self.actions_log_prob = None
        self.actor_obs = None
        self.expert_actions = None
        self.failed_envs = None
        self.dones = None
        self.actions_log_prob = torch.zeros((self.num_envs, 1)).to(self.device)

    def observe(self, expert_actions, actor_obs):
        self.actor_obs = actor_obs
        self.expert_actions = torch.from_numpy(expert_actions).to(self.device)

        # determine learner action. self.actor.sample samples an action according to the current distribution based on
        # the determined action. Noiseless action is the direct output of the neural network.
        self.actions, self.actions_log_prob = self.actor.sample(torch.from_numpy(actor_obs).to(self.device))
        if self.deterministic_policy:
            self.actions = self.actor.noiseless_action(torch.from_numpy(actor_obs).to(self.device))
        # self.actions = np.clip(self.actions.numpy(), self.env.action_space.low, self.env.action_space.high)

        return self.actions

    def step(self, value_obs, rews, dones):
        # add gathered data to the storage
        values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))
        self.storage.add_transitions(self.actor_obs, value_obs, self.actions, self.expert_actions, rews, dones, values,
                                     self.actions_log_prob)

    def update(self, actor_obs, value_obs, log_this_iteration, update, reward_sum):
        # predict terminal state reward
        last_values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))

        # compute returns, advantages, etc.
        self.storage.compute_returns(last_values.to(self.device), self.gamma, self.lam)

        # learning step
        mean_loss, mean_value_loss, mean_surrogate_loss, mean_action_log_prob_loss, mean_entropy_loss, mean_returns, \
        mean_advantages, mean_bc_loss, infos = self._train_step()
        mean_cumul_reward = reward_sum
        self.storage.clear()

        if log_this_iteration:
            self.log({**locals(), **infos, 'it': update})

        return mean_loss

    def log(self, variables, width=80, pad=28):
        self.tot_timesteps += self.num_transitions_per_env * self.num_envs
        mean_std = self.actor.distribution.std.mean()

        self.writer.add_scalar('Loss/mean_loss', variables['mean_loss'], variables['it'])
        self.writer.add_scalar('Loss/value_function', variables['mean_value_loss'], variables['it'])
        self.writer.add_scalar('Loss/surrogate', variables['mean_surrogate_loss'], variables['it'])
        self.writer.add_scalar('Loss/mean_action_log_prob_loss', variables['mean_action_log_prob_loss'], variables['it'])
        self.writer.add_scalar('Loss/mean_entropy_loss', variables['mean_entropy_loss'], variables['it'])
        self.writer.add_scalar('Loss/mean_bc_loss', variables['mean_bc_loss'], variables['it'])
        self.writer.add_scalar('Other/mean_returns', variables['mean_returns'], variables['it'])
        self.writer.add_scalar('Other/mean_cumul_returns', variables['mean_cumul_reward'], variables['it'])
        self.writer.add_scalar('Other/failed_envs', self.failed_envs, variables['it'])
        self.writer.add_scalar('Other/dones', self.dones, variables['it'])
        self.writer.add_scalar('Other/mean_advantages', variables['mean_advantages'], variables['it'])
        self.writer.add_scalar('Other/mean_noise_std', mean_std.item(), variables['it'])

    def _train_step(self):
        # this method determines the number of failed environments (= quadcopter collided with the ground)
        self.failed_envs, self.dones = self.storage.find_failed_envs()

        # for logging
        mean_loss = 0
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_action_log_prob_loss = 0
        mean_returns = 0
        mean_advantages = 0
        mean_bc_loss = 0

        # actual training loop
        for epoch in range(self.num_learning_epochs):
            for actor_obs_batch, critic_obs_batch, actions_batch, expert_actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
                    in self.batch_sampler(self.num_mini_batches):

                # define loss functions
                actions_log_prob_batch, entropy_batch = self.actor.evaluate(actor_obs_batch, actions_batch)
                value_batch = self.critic.evaluate(critic_obs_batch)
                expert_log, _ = self.actor.evaluate(actor_obs_batch, expert_actions_batch)

                l2_reg = [torch.sum(torch.square(w)) for w in self.actor.parameters() and self.critic.parameters()]
                l2_reg_norm = sum(l2_reg) / 2
                l2_reg_loss = self.l2_reg_coef * l2_reg_norm

                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))

                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                   1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                bc_loss = -expert_log.mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() \
                       + l2_reg_loss + self.bc_coef * bc_loss

                # optimization step based on loss function
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], self.max_grad_norm)
                self.optimizer.step()

                mean_loss += loss
                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy_loss += -entropy_batch.mean().item()
                mean_action_log_prob_loss += -actions_log_prob_batch.mean().item()
                mean_returns += returns_batch.mean().item()
                mean_advantages += advantages_batch.mean().item()
                mean_bc_loss += bc_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_loss /= num_updates
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_action_log_prob_loss /= num_updates
        mean_returns /= num_updates
        mean_advantages /= num_updates
        mean_bc_loss /= num_updates

        return mean_loss, mean_value_loss, mean_surrogate_loss, mean_action_log_prob_loss, mean_entropy_loss, \
               mean_returns, mean_advantages, mean_bc_loss, locals()