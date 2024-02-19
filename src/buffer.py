import torch
import numpy as np


class Buffer:

    def __init__(self, batch_size, dimension, path_length, max_actions, device):
        self.batch_size = batch_size
        self.dimension = dimension
        self.path_length = path_length
        self.max_actions = max_actions
        self.device = device
        self._init_buffers()

    def _init_buffers(self):
        self.returns = torch.zeros((self.batch_size, self.path_length), device=self.device)
        self.observations = torch.zeros((self.batch_size, self.path_length, self.dimension), device=self.device)
        self.actions_tensors = torch.zeros((self.batch_size, self.path_length,
                                            self.max_actions, self.dimension), device=self.device)
        self.actions = torch.zeros((self.batch_size, self.path_length,), device=self.device, dtype=torch.int32)
        self.log_prob_actions = torch.zeros((self.batch_size, self.path_length,), device=self.device)
        self.values = torch.zeros((self.batch_size, self.path_length,), device=self.device)

    def add_step(self, episode, step, obs, actions_tensor, action, log_prob_action, reward, value=None):
        self.observations[episode][step].copy_(obs)
        self.actions_tensors[episode][step].copy_(actions_tensor)
        self.actions[episode][step].copy_(action)
        self.log_prob_actions[episode][step].copy_(log_prob_action)
        self.returns[episode][step].copy_(reward)
        self.values[episode][step].copy_(value)

    # TODO: Vectorize
    def compute_returns(self, discount=0.99, lambda_gae=0.95, use_gae=False):
        if use_gae:
            self._gae(discount, lambda_gae)
        else:
            for step in range(self.returns.size(1)-2, -1, -1):
                self.returns[:, step] = self.returns[:, step] + self.returns[:, step+1] * discount

    def normalize_returns(self):
        var, mean = torch.var_mean(self.returns)
        epsilon = np.finfo(np.float32).eps
        self.returns = (self.returns - mean) / (torch.sqrt(var) + epsilon)

    # TODO: Vectorize
    def _gae(self, discount, lambda_gae):
        advantage = self.returns[:, -1] - self.values[:, -1]
        for step in range(self.returns.size(1)-2, -1, -1):
            td_error = self.returns[:, step] + self.values[:, step+1] * discount - self.values[:, step]
            advantage = td_error + advantage * discount * lambda_gae
            self.returns[:, step] = advantage + self.values[:, step]
