import torch.nn as nn
import torch
import gym
import numpy as np
from time import time
import pandas as pd
from collections import deque
import random
import torch.nn.functional as nnF


class FeedForwardTorch(nn.Module):
    def __init__(self, dimensions, activations, loss_fun, optimizer):
        super(FeedForwardTorch, self).__init__()

        self.layer_count = len(dimensions) - 1
        if self.layer_count == 1:
            self.ffw = nn.Linear(dimensions[0], dimensions[1])
            if activations and activations[0]:
                self.ffw = activations[0](self.ffw)
        else:
            layers = []
            if activations:
                for i in range(self.layer_count):
                    layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
                    if activations[i]:
                        layers.append(activations[i]())
            else:
                for i in range(self.layer_count):
                    layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))

            self.ffw = nn.Sequential(*layers)

        self.loss_fun = loss_fun
        self.optimizer = optimizer(self.parameters())

    def forward(self, x):
        return self.ffw(x)

    def train(self, mini_batches, stop_loss=1e-6, max_iter=1000, mode=True):
        self.ffw.train()
        # super(FeedForwardTorch, self).train(mode)
        loss = None
        if mini_batches:
            for _ in range(max_iter):
                for x, y in mini_batches:
                    self.zero_grad()
                    loss = self.loss_fun(self.__call__(x), y)
                    # loss = self.loss_fun(x, y)
                    if loss < stop_loss:
                        break
                    loss.backward()
                    self.optimizer.step()
        return loss

    def predict(self, x):
        self.ffw.eval()
        return self.__call__(x)

    def save(self, path):
        torch.save(self.ffw.state_dict(), path)

    def load(self, path):
        self.ffw.load_state_dict(torch.load(path))
        self.ffw.eval()


# implements a general deep q-learning framework


class QFun:
    def __init__(self, model, eval_switch, list_to_tensor, nparray_to_tensor, argmax):
        self.model = model
        self.eval_switch = eval_switch
        self.list_to_tensor = list_to_tensor
        self.nparray_to_tensor = nparray_to_tensor
        self.argmax = argmax


class QLearner:
    def __init__(self, env, q_fun, epsilon_decay = 0.998, epsilon_min = 0.1, gamma = 0.99):
        self.env = env
        self.Q_fun = q_fun
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_min
        self.gamma = gamma
        self._action_count = env.action_space.n

    def _get_batch(self, replay_memory, sample_size):
        # five columns: states, actions, rewards, next states and 'done' flags
        experience_sample = random.sample(replay_memory, sample_size)
        states = [record[0] for record in experience_sample]
        states_next = [record[3] for record in experience_sample]
        if self.Q_fun.list_to_tensor:
            states = self.Q_fun.list_to_tensor(states)
            states_next = self.Q_fun.list_to_tensor(states_next)
        q_vals = self.Q_fun.model.predict(states)
        q_vals_next = self.Q_fun.model.predict(states_next)

        for i in range(sample_size):
            action_i = experience_sample[i][1]
            reward_i = experience_sample[i][2]

            if experience_sample[i][4]:  # if done
                q_vals[i, action_i] = reward_i
            else:
                q_vals[i, action_i] = float(reward_i + self.gamma * max(q_vals_next[i]))

        return states, q_vals

    def run(self, episode_count = 2500,
            epsilon_start=1.0,
            replay_memory=2 ** 16,
            replay_sample_size=32,
            training_start_memory_size=64,
            stop_mean_reward=None,
            mean_reward_recency=100,
            start_episode_idx=0,
            logging=True,
            train=True,
            render=False):

        if episode_count == 0:
            return 0, None

        logs = None
        if type(logging) is pd.DataFrame:
            logs = logging
            logging = True

        loss = -1
        env, Q_fun = self.env, self.Q_fun

        if train:
            if type(replay_memory) is int:
                replay_memory = deque([], maxlen=replay_memory)
            elif type(replay_memory) is list:
                replay_memory = deque(replay_memory, maxlen=len(replay_memory))
            elif type(replay_memory) is not deque:
                raise ValueError("The replay_memory should be an integer, a list or a collections.deque object")
        elif Q_fun.eval_switch is not None:
            Q_fun.eval_switch()

        recent_rewards = [0] * mean_reward_recency
        for episode_idx in range(start_episode_idx, episode_count + start_episode_idx):
            curr_total_reward = 0
            curr_state = env.reset()
            done = False
            action_count = 0

            train_start_time = time()
            while not done:
                prev_state = curr_state

                if epsilon_start > 0 and random.random() < epsilon_start:
                    act = env.action_space.sample()
                else:
                    if Q_fun.nparray_to_tensor:
                        curr_state = Q_fun.nparray_to_tensor(curr_state)
                    act = int(Q_fun.argmax(Q_fun.model.predict(curr_state)))

                curr_state, reward, done, next_state = env.step(act)
                if render:
                    env.render();
                if train:
                    replay_memory.append((prev_state, act, reward, curr_state, done))
                    if len(replay_memory) >= training_start_memory_size:
                        loss = float(Q_fun.model.train([self._get_batch(replay_memory, replay_sample_size)], max_iter=1))
                else:
                    loss = 0
                curr_total_reward += reward
                action_count += 1
            curr_time_use = time() - train_start_time

            if epsilon_start > self.epsilon_end:
                epsilon_start *= self.epsilon_decay

            recent_rewards[(episode_idx - start_episode_idx) % mean_reward_recency] = curr_total_reward

            if episode_idx - start_episode_idx >= mean_reward_recency - 1:
                recent_mean_reward = sum(recent_rewards) / mean_reward_recency
                if stop_mean_reward is not None and recent_mean_reward > stop_mean_reward:
                    break
            else:
                recent_mean_reward = -1

            log_entry = (episode_idx, curr_total_reward, recent_mean_reward, loss, epsilon_start, action_count, curr_time_use)
            if logging:
                if logs is None:
                    logs = pd.DataFrame(columns=("Episode", "Total Reward", "Mean Reward", "Train Loss", "Epsilon", "Actions", "Training Time"))
                logs.loc[episode_idx] = log_entry
            print("Episode:{}, Total Reward: {}, Mean Reward: {:.2f}, Train Loss: {:.3f}, Epsilon: {:.5f}, Actions: {}, Training Time: {:.3f}".format(*log_entry))

        logs["Episode"] = logs["Episode"].astype(int)
        logs["Actions"] = logs["Actions"].astype(int)
        return recent_mean_reward, logs

    def train(self, episode_count = 2500, epsilon_start=1.0, replay_memory=2 ** 16, replay_sample_size=32, training_start_memory_size=64, stop_mean_reward=None, mean_reward_recency=100, start_episode_idx=0, logging=True, render=False):
        return self.run(episode_count=episode_count,
                        epsilon_start=epsilon_start,
                        replay_memory=replay_memory,
                        replay_sample_size=replay_sample_size,
                        training_start_memory_size=max(replay_sample_size, training_start_memory_size),
                        stop_mean_reward=stop_mean_reward,
                        start_episode_idx=start_episode_idx,
                        mean_reward_recency=mean_reward_recency,
                        logging=logging, train=True, render=render)

    def test(self, start_episode_idx=0, episode_count=100, logging=True, continued_learning=True, replay_memory=2 ** 16, replay_sample_size=32, render=False):
        return self.run(start_episode_idx=start_episode_idx,
                        episode_count=episode_count,
                        epsilon_start=0,
                        replay_memory=replay_memory,
                        replay_sample_size=replay_sample_size,
                        mean_reward_recency=episode_count,
                        logging=logging, train=continued_learning, render=render)


# implements the lunar lander agent

def _get_dimensions(env, hidden_layer_dimensions):
    state_dim = env.observation_space.shape[0]
    action_count = env.action_space.n
    dimensions = [state_dim] + hidden_layer_dimensions + [action_count]
    return dimensions


def _get_qfun(framework, dimensions, learning_rate, model_path=None, use_dropout=False):
    if framework.startswith("pytorch"):
        ffw = FeedForwardTorch(dimensions=dimensions,
                               activations=[(nn.BatchNorm1d if use_dropout else nn.ReLU) for _ in range(len(dimensions) - 2)] + [None],
                               loss_fun=nnF.mse_loss,
                               optimizer=lambda mode_paras: torch.optim.Adam(mode_paras, lr=learning_rate, betas=(0.9, 0.98), eps=1e-8))
        if framework.endswith("_cuda") and torch.cuda.is_available():
            device = torch.device("cuda:0")
            ffw.cuda(device)
            q_fun = QFun(model=ffw,
                         eval_switch=ffw.eval,
                         list_to_tensor=torch.cuda.FloatTensor,
                         nparray_to_tensor=lambda x: torch.from_numpy(x).to(device),
                         argmax=torch.argmax)
        else:
            q_fun = QFun(model=ffw,
                         eval_switch=ffw.eval,
                         list_to_tensor=torch.Tensor,
                         nparray_to_tensor=torch.from_numpy,
                         argmax=torch.argmax)
    else:
        raise ValueError("Framework '{}' is not supported. Use 'pytorch' or 'pytorch_cuda'".format(framework))

    if model_path:
        ffw.load(path=model_path)

    return q_fun


def train_lunar_lander(env, framework="pytorch", hidden_layer_dimensions = [128, 64],
                       use_dropout=False,
                       training_episode_count=2000,
                       alpha=1e-4, gamma=0.99,
                       epsilon_start=1.0, epsilon_decay=0.998, epsilon_min=0.0,
                       replay_memory_size=2 ** 16, replay_sample_size=32, training_start_memory_size=64,
                       mean_reward_recency=100):
    dimensions = _get_dimensions(env, hidden_layer_dimensions)
    q_fun = _get_qfun(framework, dimensions, alpha, use_dropout=use_dropout)

    lunar_lander = QLearner(env=env,
                            q_fun=q_fun,
                            epsilon_decay=epsilon_decay,
                            epsilon_min=epsilon_min,
                            gamma=gamma)

    mean_reward, logs = lunar_lander.train(episode_count=training_episode_count,
                                           epsilon_start=epsilon_start,
                                           replay_memory=replay_memory_size,
                                           replay_sample_size=replay_sample_size,
                                           training_start_memory_size=training_start_memory_size,
                                           mean_reward_recency=mean_reward_recency)


env = gym.make('LunarLander-v2')

train_lunar_lander(env, framework="pytorch", hidden_layer_dimensions = [128, 64],
                        use_dropout=False,
                        training_episode_count=2000,
                        alpha=1e-4, gamma=0.99,
                        epsilon_start=1.0, epsilon_decay=0.998, epsilon_min=0.0,
                        replay_memory_size=2 ** 16, replay_sample_size=32, training_start_memory_size=64,
                        mean_reward_recency=100)
