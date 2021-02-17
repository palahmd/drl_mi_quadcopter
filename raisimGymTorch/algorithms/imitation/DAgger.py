import torch.nn as nn
import torch
from .storage import RolloutStorage
from .behavioralCloning import BCTrainer
from imitation.algorithms import dagger
from stable_baselines3.common import policies

class ActorCritic64Policy(policies.ActorCriticPolicy)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, net_arch=[64, 64], activation_fn=nn.ReLU)

class DAggerRaisim:
    def __init__(self,
                 env,
                 actor,
                 critic,
                 num_envs,
                 num_transitions_per_env,
                 num_mini_batches,
                 num_learning_epochs,
                 save_dir,
                 beta,
                 l2_reg_weight,
                 device='cpu'):

        self.env = env
        self.actor = actor
        self.critic = critic
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.num_mini_batches = num_mini_batches
        self.num_learning_epochs = num_learning_epochs
        self.save_dir = save_dir
        self.beta = beta
        self.l2_reg_weight = l2_reg_weight
        self.device = device

        self.beta_schedule = dagger.LinearBetaSchedule(beta)
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor.obs_shape, critic.obs_shape, actor.action_shape, device)
        self.batch_sampler = self.storage.mini_batch_generator_shuffle

        # Log
        self.log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0

        #temps
        self.actions = None
        self.acttor_obs = None
        self.expert_actions = None


    def observe(self, actor_obs):
        self.actor_obs = actor_obs
        self.actions, self.actions_log_prob = self.actor.sample(torch.from_numpy(actor_obs).to(self.device))
        # self.actions = np.clip(self.actions.numpy(), self.env.action_space.low, self.env.action_space.high)
        return self.actions.cpu().numpy()

    def step(self, value_obs, expert_actions, rews, dones):
        values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))
        self.storage.add_transitions(self.actor_obs, value_obs, self.actions, expert_actions, rews, dones, values,
                                     self.actions_log_prob)

    def update(self, log_this_iteration):
        mean_value_loss, infos = self._train_step_with_behavioral_cloning()

        if log_this_iteration:
            self.log({**locals(), **infos, 'it': update})

    def _train_step_with_behavioral_cloning(self):
        mean_action_loss = 0
        for epoch in range(self.num_learning_epochs):
            for actor_obs_batch, actions_batch, expert_actions_batch in self.batch_sampler(self.num_mini_batches):

                act_log_prob_batch, entropy_batch = self.actor.evaluate(actor_obs_batch, actions_batch)
                l2_reg_norm_actor = [torch.sum(torch.square(w)) for w in self.actor.parameters()]
                l2_reg_norm_critic = [torch.sum(torch.square(w)) for w in self.critic.parameters()]
                l2_reg_norm = sum(l2_reg_norm_actor, l2_reg_norm_critic) / 2

                action_loss = (actions_batch - expert_actions_batch).pow(2).mean()
                entropy_loss = -entropy_batch.mean()
                l2_reg_loss = self.l2_reg_weight * l2_reg_norm

                loss = action_loss + entropy_loss + l2_reg_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                mean_action_loss += action_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_action_loss /= num_updates

        return mean_action_loss, locals()