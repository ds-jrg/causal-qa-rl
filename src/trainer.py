from abc import abstractmethod
import numpy as np
import torch
import wandb
import random

from environment import EnvironmentTorch
from agents import LSTMActorCriticAgent
from utils.utils_agent import bfs
from utils.utils_agent import run_beam_search
from utils.utils_agent import run_greedy_decoding
from utils.utils_agent import compute_metrics
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim


class Trainer:

    def __init__(self, train_env: EnvironmentTorch, eval_env: EnvironmentTorch, agent, device,
                 use_wandb: bool = False,
                 batch_size: int = 10,
                 eval_interval: int = 10,
                 eval_episodes: int = 10,
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
        self.eval_env = eval_env
        self.agent = agent
        self.device = device
        self.optimizer = optim.AdamW(self.agent.parameters(), lr=lr)
        self.use_wandb = use_wandb

        self.batch_size = batch_size
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes

        self.seed = seed
        self.supervised = supervised
        self.use_full_path = use_full_path
        self.supervised_ratio = supervised_ratio
        self.supervised_steps = supervised_steps
        self.supervised_batch_size = supervised_batch_size
        self.beam_search = beam_search
        self.beam_width = beam_width
        self.max_grad_norm = max_grad_norm
        self.discount = discount
        self.beta_entropy = beta_entropy
        self.normalize_returns = normalize_returns

        self.seen_paths = set()

    def train(self, steps: int = 1000):
        if self.supervised:
            self.supervised_training()
            metrics,  _ = self.evaluate()
            print('supervised: Average Return Eval = {0}'.format(metrics['accuracy']))
            if self.use_wandb:
                wandb.log({**{'avg_return_eval': metrics['accuracy'],
                              'unique_paths': 0}, **metrics})

        train_returns = []
        eval_returns = []
        # wandb.watch(self.agent, log='all', log_freq=self.eval_interval)
        for step in range(1, steps+1):
            loss, avg_return_train, entropy = self._run_batch()

            if step % self.eval_interval == 0:
                metrics,  _ = self.evaluate()
                print('step = {0}: loss = {1}'.format(step, loss))
                print('step = {0}: Average Return Train = {1}'.format(step, avg_return_train))
                print('step = {0}: Average Return Eval = {1}'.format(step, metrics['accuracy']))
                if self.use_wandb:
                    wandb.log({**{'train_loss': loss,
                                  'unique_paths': len(self.seen_paths),
                                  'entropy': entropy,
                                  'avg_return_train': avg_return_train,
                                  'avg_return_eval': metrics['accuracy']}, **metrics})
                train_returns.append(avg_return_train)
                eval_returns.append(metrics['accuracy'])

        return sum(eval_returns) / len(eval_returns)

    def evaluate(self):
        self.agent.eval()
        if self.beam_search:
            true_labels, predictions, question_candidates, _ = run_beam_search(self.agent,
                                                                               self.eval_env,
                                                                               self.device,
                                                                               self.beam_width)
        else:
            true_labels, predictions, _ = run_greedy_decoding(self.agent, self.eval_env, self.device)
            question_candidates = []

        metrics = compute_metrics(true_labels, predictions)
        return metrics, question_candidates

    @abstractmethod
    def _run_batch(self):
        pass

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
        questions = random.sample(self.train_env.questions, number_questions)
        paths = []
        for q in questions:
            path, _ = bfs(kg, kg.entity_to_id(q.cause), kg.entity_to_id(q.effect), self.train_env.max_path_len)
            if path is not None:
                paths.append((path, q.id_))
        paths = get_batches(paths)

        for step in range(self.supervised_steps):
            log_prob_actions = []
            entropy_actions = []
            for path in paths[step % len(paths)]:
                episode_log_prob_actions = []
                episode_entropy_actions = []

                agent_state = self.agent.get_initial_state(self.device)
                for idx, entity in enumerate(path[0][:-1]):
                    state_ob, state_action = self.train_env.get_state(entity, path[1])
                    if entity != path[0][idx+1]:
                        next_action = torch.tensor(kg.neighbour_ids(entity).index(path[0][idx+1]))
                    else:
                        next_action = torch.tensor(0)
                    if isinstance(self.agent, LSTMActorCriticAgent):
                        action_pred, _, agent_state = self.agent(state_ob.to(self.device),
                                                                 state_action.to(self.device),
                                                                 agent_state)
                    else:
                        action_pred, agent_state = self.agent(state_ob.to(self.device),
                                                              state_action.to(self.device),
                                                              agent_state)
                    action_pred.masked_fill_(action_pred == 0.0, float('-inf'))
                    action_prob = F.softmax(action_pred, dim=-1).cpu()
                    dist = Categorical(action_prob)

                    log_prob_action = dist.log_prob(next_action)
                    episode_log_prob_actions.append(log_prob_action)
                    episode_entropy_actions.append(dist.entropy())

                log_prob_actions.append(torch.cat(episode_log_prob_actions))
                entropy_actions.append(torch.cat(episode_entropy_actions))

            log_prob_actions = torch.stack(log_prob_actions)
            entropy_actions = torch.stack(entropy_actions)

            rewards = torch.ones_like(log_prob_actions)
            discounted_returns = torch.stack([self._discounted_reward_to_go(episode_reward.numpy())
                                              for episode_reward in rewards])

            loss = self._update_policy_supervised(discounted_returns, log_prob_actions, entropy_actions)
            print(loss)

    def _discounted_reward_to_go(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + R * self.discount
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def _normalize_returns(self, discounted_returns):
        var, mean = torch.var_mean(discounted_returns)
        epsilon = np.finfo(np.float32).eps
        return (discounted_returns - mean) / (torch.sqrt(var) + epsilon)

    def _advantages(self, discounted_returns, values):
        return discounted_returns - values

    def _reinforce_loss_with_entropy_reg(self, returns, log_probs, entropy_actions):
        if self.beta_entropy > 0:
            loss = - (returns * log_probs).sum(1).mean() - self.beta_entropy * entropy_actions.mean()
        else:
            loss = - (returns * log_probs).sum(1).mean()
        return loss

    def _grad_norm_check(self):
        total_norm = 0
        parameters = [p for p in self.agent.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm


class REINFORCETrainer(Trainer):

    def __init__(self, train_env: EnvironmentTorch, eval_env: EnvironmentTorch, agent, device,
                 use_wandb: bool = False,
                 batch_size: int = 10,
                 eval_interval: int = 10,
                 eval_episodes: int = 10,
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
        super().__init__(train_env, eval_env, agent, device,
                         use_wandb=use_wandb,
                         batch_size=batch_size,
                         eval_interval=eval_interval,
                         eval_episodes=eval_episodes,
                         lr=lr,
                         seed=seed,
                         supervised=supervised,
                         supervised_steps=supervised_steps,
                         supervised_batch_size=supervised_batch_size,
                         use_full_path=use_full_path,
                         supervised_ratio=supervised_ratio,
                         beam_search=beam_search,
                         beam_width=beam_width,
                         max_grad_norm=max_grad_norm,
                         discount=discount,
                         beta_entropy=beta_entropy,
                         normalize_returns=normalize_returns)

    def _run_batch(self):
        self.agent.train()

        log_prob_actions = []
        entropy_actions = []
        rewards = []
        total_return = 0

        for _ in range(self.batch_size):
            episode_reward = 0.0
            episode_log_prob_actions = []
            episode_entropy_actions = []
            episode_rewards = []

            state_ob, state_action = self.train_env.reset()
            agent_state = self.agent.get_initial_state(self.device)
            while True:
                action_pred, agent_state = self.agent(state_ob.to(self.device),
                                                      state_action.to(self.device),
                                                      agent_state)
                action_pred.masked_fill_(action_pred == 0.0, float('-inf'))
                action_prob = F.softmax(action_pred, dim=-1).cpu()
                dist = Categorical(action_prob)
                action = dist.sample()
                (state_ob, state_action), reward, done = self.train_env.step(action.item())

                log_prob_action = dist.log_prob(action)
                episode_log_prob_actions.append(log_prob_action)
                episode_entropy_actions.append(dist.entropy())
                episode_rewards.append(reward)

                episode_reward += reward
                if done:
                    break

            log_prob_actions.append(torch.cat(episode_log_prob_actions))
            entropy_actions.append(torch.cat(episode_entropy_actions))
            rewards.append(torch.tensor(episode_rewards, dtype=torch.float32))

            total_return += episode_reward
            self.seen_paths.add(tuple(self.train_env.current_path))

        log_prob_actions = torch.stack(log_prob_actions)
        entropy_actions = torch.stack(entropy_actions)

        rewards = torch.stack(rewards)
        discounted_returns = torch.stack([self._discounted_reward_to_go(episode_reward.numpy())
                                          for episode_reward in rewards])

        if self.normalize_returns:
            discounted_returns = self._normalize_returns(discounted_returns)

        loss = self._update_policy(discounted_returns, log_prob_actions, entropy_actions)
        return loss, total_return / self.batch_size, entropy_actions.mean().item()

    def _update_policy(self, returns, log_probs, entropy_actions):
        returns = returns.detach()
        loss = self._reinforce_loss_with_entropy_reg(returns, log_probs, entropy_actions)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item()

    def _update_policy_supervised(self, returns, log_probs, entropy_actions):
        returns = returns.detach()
        loss = self._reinforce_loss_with_entropy_reg(returns, log_probs, entropy_actions)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item()


class A2CTrainer(Trainer):

    def __init__(self, train_env: EnvironmentTorch, eval_env: EnvironmentTorch, agent, device,
                 use_wandb: bool = False,
                 batch_size: int = 10,
                 eval_interval: int = 10,
                 eval_episodes: int = 10,
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
                 use_advantage: bool = False,
                 use_gae: bool = False,
                 lambda_gae: float = 0.95):
        super().__init__(train_env, eval_env, agent, device,
                         use_wandb=use_wandb,
                         batch_size=batch_size,
                         eval_interval=eval_interval,
                         eval_episodes=eval_episodes,
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
        self.use_advantage = use_advantage
        self.use_gae = use_gae
        self.lambda_gae = lambda_gae

    def _run_batch(self):
        self.agent.train()

        log_prob_actions = []
        values = []
        entropy_actions = []
        rewards = []
        gae = []
        total_return = 0

        for _ in range(self.batch_size):
            episode_reward = 0.0
            episode_log_prob_actions = []
            episode_values = []
            episode_entropy_actions = []
            episode_rewards = []

            state_ob, state_action = self.train_env.reset()
            agent_state = self.agent.get_initial_state(self.device)
            while True:
                action_pred, value_pred, agent_state = self.agent(state_ob.to(self.device),
                                                                  state_action.to(self.device),
                                                                  agent_state)
                action_pred.masked_fill_(action_pred == 0.0, float('-inf'))
                action_prob = F.softmax(action_pred, dim=-1).cpu()
                dist = Categorical(action_prob)
                action = dist.sample()
                (state_ob, state_action), reward, done = self.train_env.step(action.item())

                log_prob_action = dist.log_prob(action)
                episode_log_prob_actions.append(log_prob_action)
                episode_values.append(value_pred.cpu())
                episode_entropy_actions.append(dist.entropy())
                episode_rewards.append(reward)

                episode_reward += reward
                if done:
                    break

            log_prob_actions.append(torch.cat(episode_log_prob_actions))
            episode_values = torch.cat(episode_values)
            values.append(episode_values)
            entropy_actions.append(torch.cat(episode_entropy_actions))

            rewards.append(torch.tensor(episode_rewards, dtype=torch.float32))
            gae.append(self._gae(episode_rewards, episode_values))

            total_return += episode_reward
            self.seen_paths.add(tuple(self.train_env.current_path))

        log_prob_actions = torch.stack(log_prob_actions)
        values = torch.stack(values)

        entropy_actions = torch.stack(entropy_actions)
        gae = torch.stack(gae)

        rewards = torch.stack(rewards)

        discounted_returns = torch.stack([self._discounted_reward_to_go(episode_reward.numpy())
                                          for episode_reward in rewards])

        if self.normalize_returns:
            discounted_returns = self._normalize_returns(discounted_returns)

        if self.use_gae:
            advantages = gae
        else:
            advantages = self._advantages(discounted_returns, values)

        loss = self._update_policy(discounted_returns, log_prob_actions, advantages, entropy_actions)
        return loss, total_return / self.batch_size, entropy_actions.mean().item()

    def _update_policy(self, returns, log_probs, advantages, entropy_actions):
        # critic_loss = F.smooth_l1_loss(returns, values)
        critic_loss = advantages.pow(2).mean()
        actor_loss = self._reinforce_loss_with_entropy_reg(advantages.detach(), log_probs, entropy_actions)
        loss = actor_loss + self.value_loss_coef * critic_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item()

    def _gae(self, rewards, values):
        advantages = []
        advantage = 0
        next_value = 0

        for r, v in zip(reversed(rewards), reversed(values)):
            td_error = r + next_value * self.discount - v
            advantage = td_error + advantage * self.discount * self.lambda_gae
            next_value = v.detach()
            advantages.insert(0, advantage)

        return torch.stack(advantages)

    def _update_policy_supervised(self, returns, log_probs, entropy_actions):
        returns = returns.detach()
        loss = self._reinforce_loss_with_entropy_reg(returns, log_probs, entropy_actions)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item()
