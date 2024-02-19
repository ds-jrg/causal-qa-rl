from copy import copy
from typing import List
import numpy as np
import torch
import wandb
import random

from environment import EnvironmentTorch
from buffer import Buffer
from utils.agent_utils import bfs
from utils.agent_utils import run_beam_search
from utils.agent_utils import run_greedy_decoding
from utils.agent_utils import compute_metrics
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim


class Trainer:

    def __init__(self, train_env: EnvironmentTorch, eval_envs: List[EnvironmentTorch], agent, device,
                 use_wandb: bool = False,
                 batch_size: int = 10,
                 eval_interval: int = 10,
                 lr: float = 1e-4,
                 seed: int = 42,
                 supervised: bool = False,
                 use_full_path: bool = True,
                 supervised_ratio: float = 0.2,
                 supervised_steps: int = 100,
                 supervised_batch_size: int = 32,
                 beam_search: bool = False,
                 beam_width: int = 5,
                 max_grad_norm: float = 0.5,
                 discount: float = 0.99,
                 beta_entropy: float = 0.01,
                 normalize_returns: bool = False):
        self.train_env = train_env
        self.eval_envs = eval_envs
        self.agent = agent
        self.device = device
        self.optimizer = optim.AdamW(self.agent.parameters(), lr=lr)
        self.use_wandb = use_wandb

        self.batch_size = batch_size
        self.discount = discount
        self.eval_interval = eval_interval

        self.seed = seed
        self.supervised = supervised
        self.use_full_path = use_full_path
        self.supervised_ratio = supervised_ratio
        self.supervised_steps = supervised_steps
        self.supervised_batch_size = supervised_batch_size
        self.beam_search = beam_search
        self.beam_width = beam_width
        self.max_grad_norm = max_grad_norm
        self.beta_entropy = beta_entropy
        self.normalize_returns = normalize_returns

        self.seen_paths = set()

    def train(self, steps: int = 1000):
        path_tables = {}
        if self.use_wandb:
            for eval_env in self.eval_envs:
                columns = ["Question: " + question.question + "\n" + "Answer:" + str(question.binary_answer)
                           for question in eval_env.questions[:20]]
                path_table = wandb.Table(columns=columns)
                path_tables['path_table_' + eval_env.dataset_name] = path_table
            # wandb.watch(self.agent, log='all', log_freq=self.eval_interval)

        if self.supervised:
            loss, _, entropy = self.supervised_training()
            metrics, conf_matrices, visited_nodes = self.evaluate(0, path_tables)
            if self.use_wandb:
                wandb.log({**{'train_loss': loss,
                              'unique_paths': len(self.seen_paths),
                              'entropy': entropy},
                          **metrics, **visited_nodes, **conf_matrices})

        self.buffer = Buffer(self.batch_size, 2*self.train_env.graph.num_entity_dimensions,
                             self.train_env.max_path_len, self.train_env.max_actions, self.device)

        for step in range(1, steps+1):
            loss, avg_return_train, entropy = self._run_batch()

            if step % self.eval_interval == 0:
                print(f'step = {step}: loss = {loss}')
                print(f'step = {step}: Average Return Train = {avg_return_train}')
                metrics, conf_matrices, visited_nodes = self.evaluate(step, path_tables)

                if self.use_wandb:
                    wandb.log({**{'train_loss': loss,
                                  'unique_paths': len(self.seen_paths),
                                  'entropy': entropy,
                                  'avg_return_train': avg_return_train},
                               **metrics, **visited_nodes, **conf_matrices})

    def evaluate(self, step, path_tables=None):
        self.agent.eval()
        metrics = {}
        visited_nodes = {}
        conf_matrices = {}

        for eval_env in self.eval_envs:
            if self.beam_search:
                # true labels, predictions, question_candidates, number of visited nodes
                t, p, q, n = run_beam_search(self.agent, eval_env, self.device, self.beam_width)
            else:
                # true labels, predictions, question_candidates, number of visited nodes
                t, p, q, n = run_greedy_decoding(self.agent, eval_env, self.device)

            tmp_metrics = compute_metrics(t, p)
            acc = tmp_metrics['accuracy']
            print(f'step = {step}: Average Return Eval - {eval_env.dataset_name} = {acc}')

            for key, val in tmp_metrics.items():
                metrics[key + '_' + eval_env.dataset_name] = val
            visited_nodes['visited_nodes_' + eval_env.dataset_name] = n

            # TODO: Refactor into a wandb logger class
            if self.use_wandb:
                conf_matrix = wandb.plot.confusion_matrix(probs=None, y_true=t, preds=p,
                                                          class_names=['False', 'True'])
                conf_matrices['conf_mat_' + eval_env.dataset_name] = conf_matrix

                candidates = [str((c[0].path, round(float(np.exp(c[0].prob)), 2)) if len(c) > 0 else ([], 0))
                              for c in q[:20]]
                table_key = 'path_table_' + eval_env.dataset_name
                path_tables[table_key].add_data(*candidates)
                path_tables[table_key] = copy(path_tables[table_key])

        return metrics, conf_matrices, visited_nodes

    def _run_batch(self):
        self.agent.train()

        total_return = 0

        for episode in range(self.batch_size):
            state_ob, state_action = self.train_env.reset()
            agent_state = self.agent.get_initial_state(self.device)
            step = 0
            episode_finished = False
            while not episode_finished:
                with torch.no_grad():
                    action_pred, value, agent_state = self.agent(state_ob.view(1, 1, -1).to(self.device),
                                                                 state_action.view(1, 1, *state_action.shape).to(self.device),
                                                                 agent_state)
                action_pred.masked_fill_(action_pred == 0.0, float('-inf'))
                action_prob = F.softmax(action_pred, dim=-1)
                dist = Categorical(action_prob)
                action = dist.sample()
                log_prob_action = dist.log_prob(action)

                prev_state_ob, prev_state_action = state_ob, state_action
                (state_ob, state_action), reward, episode_finished = self.train_env.step(action.item())
                self.buffer.add_step(episode, step, prev_state_ob, prev_state_action, action,
                                     log_prob_action, torch.tensor(reward), value if value is not None else torch.tensor(0))

                total_return += reward
                step += 1

            self.seen_paths.add(tuple(self.train_env.current_path))

        loss, entropy = self._update_policy(self.buffer, self.batch_size)
        return loss, total_return / self.batch_size, entropy

    def supervised_training(self):
        def get_batches(x):
            return [x[i:min(i+self.supervised_batch_size, len(x))]
                    for i in range(0, len(x), self.supervised_batch_size)]

        # debug
        def print_paths(x):
            for p in paths:
                print([kg.id_to_entity(e) for e in p[0]])

        self.agent.train()
        kg = self.train_env.graph

        number_questions = int(self.supervised_ratio * len(self.train_env.questions))
        if number_questions % self.supervised_batch_size != 0:
            number_questions += (self.supervised_batch_size - (number_questions % self.supervised_batch_size))
        questions = random.sample(self.train_env.questions, min(number_questions, len(self.train_env.questions)))

        paths = []
        for q in questions:
            path, _ = bfs(kg, kg.entity_to_id(q.cause),
                          kg.entity_to_id(q.effect),
                          self.train_env.max_path_len,
                          self.train_env.max_actions)
            if path is not None:
                paths.append((path, q.id_))
        paths = get_batches(paths)

        supervised_buffer = Buffer(self.supervised_batch_size, 2*self.train_env.graph.num_entity_dimensions,
                                   self.train_env.max_path_len, self.train_env.max_actions, self.device)

        for batch in range(self.supervised_steps):
            for episode_idx, episode in enumerate(paths[batch % len(paths)]):
                for step, entity in enumerate(episode[0][:-1]):
                    state_ob, state_action = self.train_env.get_state(entity, episode[1])
                    # Check if we should use the stop action
                    if entity != episode[0][step+1]:
                        action = torch.tensor(kg.neighbour_ids(entity).index(episode[0][step+1]))
                    else:
                        action = torch.tensor(0)
                    supervised_buffer.add_step(episode_idx, step, state_ob, state_action,
                                               action, torch.tensor(0), torch.tensor(1.0), torch.tensor(0.0))

            loss, entropy = self._update_policy_supervised(supervised_buffer, self.supervised_batch_size)
        return loss, _, entropy

    def _update_policy(self, buffer, batch_size):
        buffer.compute_returns(discount=self.discount)
        if self.normalize_returns:
            self.buffer.normalize_returns()

        agent_state = self.agent.get_initial_state(self.device, batch_size)
        action_pred, _, _ = self.agent(buffer.observations, buffer.actions_tensors, agent_state)
        action_pred.masked_fill_(action_pred == 0.0, float('-inf'))
        action_prob = F.softmax(action_pred, dim=-1)
        dist = Categorical(action_prob)
        log_probs_action = dist.log_prob(buffer.actions)

        entropy = dist.entropy().mean()
        loss = self._reinforce_loss_with_entropy_reg(buffer.returns, log_probs_action, entropy)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item(), entropy.item()

    def _update_policy_supervised(self, buffer, batch_size):
        buffer.compute_returns(discount=self.discount)
        if self.normalize_returns:
            self.buffer.normalize_returns()

        agent_state = self.agent.get_initial_state(self.device, batch_size)
        action_pred, _, _ = self.agent(buffer.observations, buffer.actions_tensors, agent_state)
        action_pred.masked_fill_(action_pred == 0.0, float('-inf'))
        action_prob = F.softmax(action_pred, dim=-1)
        dist = Categorical(action_prob)
        log_probs_action = dist.log_prob(buffer.actions)

        entropy = dist.entropy().mean()
        loss = self._reinforce_loss_with_entropy_reg(buffer.returns, log_probs_action, entropy)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item(), entropy.item()

    def _reinforce_loss_with_entropy_reg(self, returns, log_probs, entropy_actions):
        return - (returns * log_probs).sum(1).mean() - self.beta_entropy * entropy_actions


class A2CTrainer(Trainer):

    def __init__(self, train_env: EnvironmentTorch, eval_envs: List[EnvironmentTorch], agent, device,
                 use_wandb: bool = False,
                 batch_size: int = 10,
                 eval_interval: int = 10,
                 lr: float = 1e-4,
                 seed: int = 42,
                 supervised: bool = False,
                 use_full_path: bool = True,
                 supervised_ratio: float = 0.2,
                 supervised_steps: int = 100,
                 supervised_batch_size: int = 32,
                 beam_search: bool = False,
                 beam_width: int = 5,
                 max_grad_norm: float = 0.5,
                 discount: float = 0.99,
                 beta_entropy: float = 0.01,
                 normalize_returns: bool = False,
                 value_loss_coef: float = 0.5,
                 use_gae: bool = False,
                 lambda_gae: float = 0.95):
        super().__init__(train_env, eval_envs, agent, device,
                         use_wandb=use_wandb,
                         batch_size=batch_size,
                         eval_interval=eval_interval,
                         lr=lr,
                         seed=seed,
                         supervised=supervised,
                         use_full_path=use_full_path,
                         supervised_ratio=supervised_ratio,
                         supervised_steps=supervised_steps,
                         supervised_batch_size=supervised_batch_size,
                         beam_search=beam_search,
                         beam_width=beam_width,
                         max_grad_norm=max_grad_norm,
                         discount=discount,
                         beta_entropy=beta_entropy,
                         normalize_returns=normalize_returns)

        self.value_loss_coef = value_loss_coef
        self.use_gae = use_gae
        self.lambda_gae = lambda_gae

    def _update_policy(self, buffer, batch_size):
        buffer.compute_returns(discount=self.discount, lambda_gae=self.lambda_gae, use_gae=True)
        if self.normalize_returns:
            self.buffer.normalize_returns()

        agent_state = self.agent.get_initial_state(self.device, batch_size)
        action_pred, values, _ = self.agent(buffer.observations, buffer.actions_tensors, agent_state)
        action_pred.masked_fill_(action_pred == 0.0, float('-inf'))
        action_prob = F.softmax(action_pred, dim=-1)
        dist = Categorical(action_prob)
        log_probs_action = dist.log_prob(buffer.actions)
        entropy = dist.entropy().mean()

        advantages = buffer.returns - values
        actor_loss = self._reinforce_loss_with_entropy_reg(advantages.detach(), log_probs_action, entropy)
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + self.value_loss_coef * critic_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item(), entropy.item()


class PPOTrainer(A2CTrainer):

    def __init__(self, train_env: EnvironmentTorch, eval_envs: List[EnvironmentTorch], agent, device,
                 use_wandb: bool = False,
                 batch_size: int = 10,
                 eval_interval: int = 10,
                 lr: float = 1e-4,
                 seed: int = 42,
                 supervised: bool = False,
                 use_full_path: bool = True,
                 supervised_ratio: float = 0.2,
                 supervised_steps: int = 100,
                 supervised_batch_size: int = 32,
                 beam_search: bool = False,
                 beam_width: int = 5,
                 max_grad_norm: float = 0.5,
                 discount: float = 0.99,
                 beta_entropy: float = 0.01,
                 normalize_returns: bool = False,
                 value_loss_coef: float = 0.5,
                 use_gae: bool = False,
                 lambda_gae: float = 0.95,
                 ppo_epochs: int = 2,
                 ppo_batch_size: int = 2,
                 ppo_clip: float = 0.2,
                 clip_critic: bool = False):
        super().__init__(train_env, eval_envs, agent, device,
                         use_wandb=use_wandb,
                         batch_size=batch_size,
                         eval_interval=eval_interval,
                         lr=lr,
                         seed=seed,
                         supervised=supervised,
                         use_full_path=use_full_path,
                         supervised_ratio=supervised_ratio,
                         supervised_steps=supervised_steps,
                         supervised_batch_size=supervised_batch_size,
                         beam_search=beam_search,
                         beam_width=beam_width,
                         max_grad_norm=max_grad_norm,
                         discount=discount,
                         beta_entropy=beta_entropy,
                         normalize_returns=normalize_returns)

        self.value_loss_coef = value_loss_coef
        self.use_gae = use_gae
        self.lambda_gae = lambda_gae
        self.ppo_epochs = ppo_epochs
        self.ppo_batch_size = ppo_batch_size
        self.ppo_clip = ppo_clip
        self.clip_critic = clip_critic

    # adapted from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py
    def _update_policy(self, buffer, batch_size):
        buffer.compute_returns(discount=self.discount, lambda_gae=self.lambda_gae, use_gae=True)
        advantages = buffer.returns - buffer.values

        # if self.normalize_returns:
        #    self.buffer.normalize_returns()
        overall_loss = 0
        overall_entropy = 0
        for _ in range(self.ppo_epochs):
            bz = batch_size // self.ppo_batch_size
            perm = torch.randperm(batch_size)
            log_probs_round = buffer.log_prob_actions[perm]
            obs_round = buffer.observations[perm]
            actions_tensors_round = buffer.actions_tensors[perm]
            actions_round = buffer.actions[perm]
            values_round = buffer.values[perm]
            returns_round = buffer.returns[perm]
            advantages_round = advantages[perm]

            log_probs_batches = torch.tensor_split(log_probs_round, bz)
            obs_batches = torch.tensor_split(obs_round, bz)
            actions_tensors_batches = torch.tensor_split(actions_tensors_round, bz)
            actions_batches = torch.tensor_split(actions_round, bz)
            values_batches = torch.tensor_split(values_round, bz)
            returns_batches = torch.tensor_split(returns_round, bz)
            advantages_batches = torch.tensor_split(advantages_round, bz)
            for batch in range(bz):
                agent_state = self.agent.get_initial_state(self.device, self.ppo_batch_size)
                action_pred, values, _ = self.agent(obs_batches[batch], actions_tensors_batches[batch], agent_state)
                action_pred.masked_fill_(action_pred == 0.0, float('-inf'))
                action_prob = F.softmax(action_pred, dim=-1)
                dist = Categorical(action_prob)
                log_probs = dist.log_prob(actions_batches[batch])
                entropy_actions = dist.entropy().mean()

                ratio = torch.exp(log_probs - log_probs_batches[batch])
                surr1 = ratio * advantages_batches[batch]
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantages_batches[batch]
                actor_loss = - torch.min(surr1, surr2).mean()

                if self.clip_critic:
                    critic_pred_clipped = values_batches[batch] + \
                        (values - values_batches[batch]).clamp(-self.ppo_clip, self.ppo_clip)
                    critic_losses = (values - returns_batches[batch]).pow(2)
                    critic_losses_clipped = (critic_pred_clipped - returns_batches[batch]).pow(2)
                    critic_loss = 0.5 * torch.max(critic_losses, critic_losses_clipped).mean()
                else:
                    critic_loss = 0.5 * (returns_batches[batch] - values).pow(2).mean()

                loss = actor_loss + self.value_loss_coef * critic_loss - entropy_actions * self.beta_entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                overall_loss += loss.item()
                overall_entropy += entropy_actions.item()

        return overall_loss, overall_entropy
