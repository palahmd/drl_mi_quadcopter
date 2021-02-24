import torch
import torch.optim as optim
import numpy as np
from .storage import RolloutStorage
from imitation.algorithms import dagger

class DAgger:
    def __init__(self,
                 actor,
                 critic,
                 num_envs,
                 num_transitions_per_env,
                 num_mini_batches,
                 num_learning_epochs,
                 beta,
                 l2_reg_weight=0.0,
                 entropy_weight=0.0,
                 learning_rate = 0.01,
                 device='cpu'):

        # Environment parameters
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.num_mini_batches = num_mini_batches
        self.num_learning_epochs = num_learning_epochs
        self.round_num = 0

        # DAgger components and parameters
        self.beta_schedule = dagger.LinearBetaSchedule(beta)
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor.obs_shape, critic.obs_shape, actor.action_shape, device)
        self.batch_sampler = self.storage.mini_batch_generator_inorder
        self.actor = actor
        self.critic = critic
        self.beta_goal = beta
        self.beta = 1
        self.l2_reg_weight = l2_reg_weight
        self.entropy_weight = entropy_weight
        self.device = device
        self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=learning_rate)

        # Log
        #self.log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
        #self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0

        # temps
        self.actions = None
        self.acttor_obs = None
        self.expert_actions = None


    def observe(self, actor_obs, expert_actions):
        self.actor_obs = actor_obs

        # reduce beta while policy is learning 
        if self.beta > self.beta_goal:
            self.beta -= 0.05 * self.round_num
            self.round_num += 1

        # take expert action with beta and policy action with (1-beta) prob.
        if np.random.uniform(0, 1) > self.beta:
            self.actions, self.actions_log_prob = self.actor.sample(torch.from_numpy(actor_obs).to(self.device))
        else:
            self.actions = torch.from_numpy(expert_actions).to(self.device)
            self.actions_log_prob = torch.ones(1)

        return self.actions.cpu().numpy()

    def step(self, value_obs, expert_actions, rews, dones):
        values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))
        expert_actions = torch.from_numpy(expert_actions)
        self.storage.add_transitions(self.actor_obs, value_obs, self.actions, expert_actions, rews, dones, values,
                                     self.actions_log_prob)

    def update(self, log_this_iteration=False):
        mean_value_loss, infos = self._train_step_with_behavioral_cloning()

        if log_this_iteration:
            self.log({**locals(), **infos, 'it': update})

        self.storage.clear()
        return mean_value_loss

    def _train_step_with_behavioral_cloning(self):
        mean_action_loss = 0
        for epoch in range(self.num_learning_epochs):
            for actor_obs_batch, expert_obs_batch, critic_obs_batch, actions_batch, expert_actions_batch, values_batch, \
                advantages_batch, returns_batch, old_actions_log_prob_batch \
                    in self.batch_sampler(self.num_mini_batches):

                act_log_prob_batch, entropy_batch = self.actor.evaluate(actor_obs_batch, actions_batch)
                #l2_reg_norm_actor = [torch.sum(torch.square(w)) for w in self.actor.parameters()]
                #l2_reg_norm_critic = [torch.sum(torch.square(w)) for w in self.critic.parameters()]
                #l2_reg_norm = (l2_reg_norm_actor + l2_reg_norm_critic) / 2

                action_loss = (actions_batch - expert_actions_batch).pow(2).mean()
                entropy_loss = self.entropy_weight*-entropy_batch.mean()
                l2_reg_loss = self.l2_reg_weight #* l2_reg_norm

                loss = action_loss + entropy_loss + l2_reg_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                mean_action_loss += action_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_action_loss /= num_updates

        return mean_action_loss, locals()

