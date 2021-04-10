from .dagger_storage import RolloutStorage
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import numpy as np
import os


class DAgger:
    def __init__(self,
                 actor,
                 critic,
                 act_dim,
                 num_envs,
                 num_transitions_per_env,
                 num_mini_batches,
                 num_learning_epochs,
                 beta,
                 log_dir,
                 use_lr_scheduler=True,
                 min_lr=0.001,
                 max_lr=0.01,
                 last_update=0,
                 l2_reg_weight=0.001,
                 entropy_weight=0.001,
                 beta_scheduler=0.0005,
                 deterministic_policy=False,
                 device='cpu',
                 shuffle_batch=True):

        # Environment parameters
        self.act_dim = act_dim
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env

        # DAgger components
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor.obs_shape, critic.obs_shape,
                                      actor.action_shape, device)
        self.actor = actor
        self.critic = critic
        self.device = device
        self.deterministic_policy = deterministic_policy

        if shuffle_batch:
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        else:
            self.batch_sampler = self.storage.mini_batch_generator_inorder

        # Training parameters
        self.num_mini_batches = num_mini_batches
        self.num_learning_epochs = num_learning_epochs
        self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=min_lr)
        self.use_lr_scheduler = use_lr_scheduler

        if self.use_lr_scheduler == True:
            self.min_lr = min_lr
            self.max_lr = max_lr
            self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.min_lr, cycle_momentum=False,
                                                         max_lr=self.max_lr, step_size_up=2*self.num_mini_batches,
                                                         last_epoch=-1, verbose=False)

            if last_update != 0:
                self.scheduler.step(epoch=last_update*num_learning_epochs*num_mini_batches)
        else:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=1)


        self.beta_goal = beta
        self.beta = 1
        self.beta_scheduler = beta_scheduler
        self.l2_reg_weight = l2_reg_weight
        self.entropy_weight = entropy_weight

        # Log
        self.log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0

        # temps
        self.learner_actions = None
        self.actor_obs = None
        self.expert_actions = None
        self.actions = np.zeros((self.num_envs, self.act_dim), dtype="float32")
        self.expert_chosen = torch.zeros((self.num_envs, 1), dtype=bool).to(self.device)
        self.learner_actions_log_prob = torch.zeros((self.num_envs, 1)).to(self.device)

    def observe(self, actor_obs, expert_actions, env_helper):
        self.actor_obs = actor_obs

        # set expert action and calculate leraner action
        self.expert_actions = torch.from_numpy(expert_actions).to(self.device)
        self.learner_actions = self.actor.noiseless_action(torch.from_numpy(actor_obs).to(self.device))
        #self.learner_actions, self.learner_actions_log_prob = self.actor.sample(torch.from_numpy(actor_obs).to(self.device))

        # take expert action with beta prob. and policy action with (1-beta) prob.
        self.choose_action_per_env()
        for i in range(0, len(self.expert_chosen)):
            if self.expert_chosen[i][0]:
                self.actions[i][:] = expert_actions[i][:]
            else:
                self.actions[i][:] = env_helper.limit_action(self.learner_actions[i][:])

        return self.actions

    def step(self, obs, rews, dones):
        values = self.critic.predict(torch.from_numpy(obs).to(self.device))
        self.storage.add_transitions(self.actor_obs, obs, self.learner_actions, self.expert_actions, rews, dones,
                                     values,
                                     self.learner_actions_log_prob)

    def update(self, obs, log_this_iteration, update):
        last_values = self.critic.predict(torch.from_numpy(obs).to(self.device))

        # calculate logging variables
        self.storage.compute_returns(last_values.to(self.device), 0.99, 0.95)
        mean_loss, mean_action_loss, mean_action_log_prob_loss, mean_l2_reg_loss, mean_entropy_loss, \
        mean_returns, mean_advantages, mean_value_loss, mean_values, infos = self._train_step_with_behavioral_cloning()

        # calculate beta for the next iteration
        self.adjust_beta()

        if log_this_iteration:
            self.log({**locals(), **infos, 'it': update})

        # clear storage for the next iteration
        self.storage.clear()

        return mean_loss, mean_action_loss, mean_action_log_prob_loss, mean_value_loss

    def log(self, variables, width=80, pad=28):
        self.tot_timesteps += self.num_transitions_per_env * self.num_envs
        mean_std = self.actor.distribution.std.mean()

        self.writer.add_scalar('Loss/mean_loss', variables['mean_loss'], variables['it'])
        self.writer.add_scalar('Loss/action_loss', variables['mean_action_loss'], variables['it'])
        self.writer.add_scalar('Loss/action_log_prob_loss', variables['mean_action_log_prob_loss'], variables['it'])
        self.writer.add_scalar('Loss/mean_entropy_loss', variables['mean_entropy_loss'], variables['it'])
        self.writer.add_scalar('Loss/mean_l2_reg_loss', variables['mean_l2_reg_loss'], variables['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), variables['it'])
        self.writer.add_scalar('Critic/mean_value_loss', variables['mean_value_loss'], variables['it'])
        self.writer.add_scalar('Critic/mean_returns', variables['mean_returns'], variables['it'])
        self.writer.add_scalar('Critic/mean_advantages', variables['mean_advantages'], variables['it'])
        self.writer.add_scalar('Critic/mean_values', variables['mean_values'], variables['it'])
        self.writer.add_scalar('Critic/mean_value_loss', variables['mean_value_loss'], variables['it'])

    def choose_action_per_env(self):
        # choose expert action with beta probability
        for i in range(0, len(self.expert_actions)):
            if np.random.uniform(0, 1) > self.beta:
                self.expert_chosen[i][0] = False
            else:
                self.expert_chosen[i][0] = True

    def adjust_beta(self):

        if self.beta <= self.beta_goal:
            self.beta = self.beta_goal
            self.beta_scheduler = -abs(self.beta_scheduler)
            if self.use_lr_scheduler:
                self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.min_lr/2, cycle_momentum=False
                                                         , max_lr=self.max_lr/2, step_size_up=2*self.num_mini_batches,
                                                         last_epoch=-1, verbose=False)

        if self.beta >= (1):
            self.beta = 1
            self.beta_scheduler = abs(self.beta_scheduler)
            if self.use_lr_scheduler:
                self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.min_lr/4, cycle_momentum=False
                                                         , max_lr=self.max_lr/4, step_size_up=2*self.num_mini_batches,
                                                         last_epoch=-1, verbose=False)

        self.beta -= self.beta_scheduler

        #if self.beta > self.beta_goal:
        #    self.beta -= self.beta_scheduler

    """ Main training: rolling out storage and training the learner with one-step behavioral cloning """

    def _train_step_with_behavioral_cloning(self):
        mean_loss = 0
        mean_action_loss = 0
        mean_action_log_prob_loss = 0
        mean_entropy_loss = 0
        mean_l2_reg_loss = 0
        mean_returns = 0
        mean_advantages = 0
        mean_value_loss = 0
        mean_values = 0
        test = 0
        for epoch in range(self.num_learning_epochs):
            for actor_obs_batch, expert_obs_batch, critic_obs_batch, actions_batch, expert_actions_batch, target_values_batch, \
                advantages_batch, returns_batch, old_actions_log_prob_batch \
                    in self.batch_sampler(self.num_mini_batches):

                act_log_prob_batch, entropy_batch = self.actor.evaluate(actor_obs_batch, expert_actions_batch)
                new_actions_batch = self.actor.noiseless_action(actor_obs_batch).to(self.device)

                l2_reg = [torch.sum(torch.square(w)) for w in self.actor.parameters() and self.critic.parameters()]
                l2_reg_norm = sum(l2_reg) / 2

                values_batch = self.critic.evaluate(critic_obs_batch)
                value_loss = (values_batch - returns_batch).pow(2).mean()

                action_loss = (new_actions_batch - expert_actions_batch).pow(2).mean()

                action_log_prob_loss = -act_log_prob_batch.mean()
                entropy_loss = self.entropy_weight * -entropy_batch.mean()
                l2_reg_loss = self.l2_reg_weight * l2_reg_norm

                if not self.deterministic_policy:
                    act_log_prob_batch, entropy_batch = self.actor.evaluate(actor_obs_batch, expert_actions_batch)
                    loss = action_log_prob_loss + entropy_loss + l2_reg_loss
                else:
                    loss = action_loss + entropy_loss + l2_reg_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                mean_loss += loss.item()
                mean_action_loss += action_loss.item()
                mean_action_log_prob_loss += action_log_prob_loss.item()
                mean_entropy_loss += -entropy_batch.mean().item()
                mean_l2_reg_loss += l2_reg_norm.item()
                mean_returns += returns_batch.mean().item()
                mean_advantages += advantages_batch.mean().item()
                mean_value_loss += value_loss.item()
                mean_values = values_batch.mean().item()
                test += 1

        print(test)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_loss /= num_updates
        mean_action_loss /= num_updates
        mean_action_log_prob_loss /= num_updates
        mean_l2_reg_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_returns /= num_updates
        mean_advantages /= num_updates
        mean_value_loss /= num_updates
        mean_values /= num_updates

        return mean_loss, mean_action_loss, mean_action_log_prob_loss, mean_l2_reg_loss, mean_entropy_loss, \
               mean_returns, mean_advantages, mean_value_loss, mean_values, locals()
