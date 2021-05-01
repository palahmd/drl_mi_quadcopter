import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal


class Actor:
    def __init__(self, architecture, distribution, device='cpu', shared_nets=False):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        self.architecture.to(device)
        self.distribution.to(device)
        self.device = device
        self.shared_nets = shared_nets

    def sample(self, obs):
        if self.shared_nets:
            logits = self.architecture.actor_net(obs)
        else:
            logits = self.architecture.architecture(obs)
        actions, log_prob = self.distribution.sample(logits)
        return actions.cpu().detach(), log_prob.cpu().detach()

    def evaluate(self, obs, actions):
        if self.shared_nets:
            action_mean = self.architecture.actor_net(obs)
        else:
            action_mean = self.architecture.architecture(obs)
        return self.distribution.evaluate(obs, action_mean, actions)

    def parameters(self):
        return [*self.architecture.parameters(), *self.distribution.parameters()]

    # TODO: try cpu().detach()
    def noiseless_action(self, obs):
        if self.shared_nets:
            return self.architecture.actor_net(obs).cpu().detach()
        else:
            return self.architecture.architecture(obs).cpu().detach()

    def save_deterministic_graph(self, file_name, example_input, device='cpu'):
        if self.shared_nets:
            transferred_graph = torch.jit.trace(self.architecture.actor_net.to(device), example_input)
            torch.jit.save(transferred_graph, file_name)
            self.architecture.actor_net.to(self.device)
        else:
            transferred_graph = torch.jit.trace(self.architecture.architecture.to(device), example_input)
            torch.jit.save(transferred_graph, file_name)
            self.architecture.architecture.to(self.device)

    def deterministic_parameters(self):
        return self.architecture.parameters()

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        if self.shared_nets:
            return self.architecture.actor_output_shape
        else:
            return self.architecture.output_shape


class Critic:
    def __init__(self, architecture, device='cpu', shared_nets=False):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)
        self.shared_nets = shared_nets

    def predict(self, obs):
        if self.shared_nets:
            return self.architecture.critic_net(obs).detach()
        else:
            return self.architecture.architecture(obs).detach()

    def evaluate(self, obs):
        if self.shared_nets:
            return self.architecture.critic_net(obs)
        else:
            return self.architecture.architecture(obs)

    def parameters(self):
        if self.shared_nets:
            return []
        else:
            return self.architecture.parameters()

    @property
    def obs_shape(self):
        return self.architecture.input_shape

class sharedBaseNetMLP(nn.Module):
    """ For a [n, n] Neural Network"""
    def __init__(self, base_shape, actor_shape, critic_shape, activation_fn, input_size, output_size):
        super(sharedBaseNetMLP, self).__init__()

        self.activation_fn = activation_fn

        base_module = [nn.Linear(input_size, base_shape[0]), self.activation_fn()]
        actor_modules = [nn.Linear(base_shape[0], actor_shape[1]), self.activation_fn()]
        critic_modules = [nn.Linear(base_shape[0], critic_shape[1]), self.activation_fn()]
        actor_modules.append(nn.Linear(actor_shape[-1], output_size[0]))
        critic_modules.append(nn.Linear(critic_shape[-1], output_size[1]))
        actor_scale = [np.sqrt(2), np.sqrt(2)]
        critic_scale = [np.sqrt(2), np.sqrt(2)]

        self.actor_net = nn.Sequential(*base_module, *actor_modules)
        self.critic_net = nn.Sequential(*base_module, *critic_modules)
        actor_scale.append(np.sqrt(2))
        critic_scale.append(np.sqrt(2))

        MLP.init_weights(self.actor_net, actor_scale)
        MLP.init_weights(self.critic_net, critic_scale)
        self.input_shape = [input_size]
        self.actor_output_shape = [output_size[0]]
        self.critic_output_shape = [output_size[1]]



class MLP(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size):
        super(MLP, self).__init__()
        self.activation_fn = actionvation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        #modules.append(nn.Identity())
        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, init_std):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        self.std = nn.Parameter(init_std * torch.ones(dim))
        self.distribution = None

    def sample(self, logits):
        self.distribution = Normal(logits, self.std.reshape(self.dim))

        samples = self.distribution.sample()
        log_prob = self.distribution.log_prob(samples).sum(dim=1)

        return samples, log_prob

    def evaluate(self, inputs, logits, outputs):
        distribution = Normal(logits, self.std.reshape(self.dim))

        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        # TODO: return action_mean/input
        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std
