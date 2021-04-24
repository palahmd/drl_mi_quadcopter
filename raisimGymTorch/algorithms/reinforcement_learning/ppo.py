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
                 use_clipped_value_loss=True,
                 log_dir='run',
                 device='cpu',
                 shuffle_batch=True,
                 deterministic_policy=False):

        # PPO components
        self.actor = actor
        self.critic = critic
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor.obs_shape, critic.obs_shape, actor.action_shape, device)

        if shuffle_batch:
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        else:
            self.batch_sampler = self.storage.mini_batch_generator_inorder

        self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=learning_rate)
        self.device = device

        # env parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

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

        # Log
        self.log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0

        # temps
        self.actions = None
        self.actions_log_prob = None
        self.actor_obs = None
        self.actions_log_prob = torch.zeros((self.num_envs, 1)).to(self.device)

    def observe(self, actor_obs):
        self.actor_obs = actor_obs
        self.actions = self.actor.noiseless_action(torch.from_numpy(actor_obs).to(self.device))
        # self.actions = np.clip(self.actions.numpy(), self.env.action_space.low, self.env.action_space.high)
        return self.actions

    def step(self, value_obs, rews, dones):
        values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))
        self.storage.add_transitions(self.actor_obs, value_obs, self.actions, rews, dones, values,
                                     self.actions_log_prob)

    def update(self, actor_obs, value_obs, log_this_iteration, update):
        last_values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))

        # Learning step
        self.storage.compute_returns(last_values.to(self.device), self.gamma, self.lam)
        mean_loss, mean_value_loss, mean_surrogate_loss, mean_action_log_prob_loss, mean_entropy_loss, mean_returns, \
        mean_advantages, infos = self._train_step()
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
        self.writer.add_scalar('mean_returns', variables['mean_returns'], variables['it'])
        self.writer.add_scalar('mean_advantages', variables['mean_advantages'], variables['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), variables['it'])

    def _train_step(self):
        mean_loss = 0
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_action_log_prob_loss = 0
        mean_returns = 0
        mean_advantages = 0
        for epoch in range(self.num_learning_epochs):
            for actor_obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
                    in self.batch_sampler(self.num_mini_batches):

                actions_log_prob_batch, entropy_batch = self.actor.evaluate(actor_obs_batch, actions_batch)
                value_batch = self.critic.evaluate(critic_obs_batch)

                # Surrogate loss
                if self.deterministic_policy:
                    new_actions_batch = self.actor.noiseless_action(actor_obs_batch)
                    ratio = torch.mean(new_actions_batch, dim=1) / torch.mean(actions_batch, dim=1)
                else:
                    ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))

                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                   1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                l2_reg = [torch.sum(torch.square(w)) for w in self.actor.parameters() and self.critic.parameters()]
                l2_reg_norm = sum(l2_reg) / 2
                l2_reg_loss = l2_reg_norm * 0.005

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + \
                       l2_reg_loss

                # Gradient step
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


        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_loss /= num_updates
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_action_log_prob_loss /= num_updates
        mean_returns /= num_updates
        mean_advantages /= num_updates

        return mean_loss, mean_value_loss, mean_surrogate_loss, mean_action_log_prob_loss, mean_entropy_loss, \
               mean_returns, mean_advantages, locals()
