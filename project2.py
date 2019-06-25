import torch.nn as nn
import torch
import gym
import numpy as np
from time import time
import pandas as pd
from collections import deque
import random
import torch.nn.functional as nnF


class FeedForwardNetwork(nn.Module):
    def __init__(self, dimensions, loss_fun, optimizer):
        super(FeedForwardNetwork, self).__init__()

        self.layer_count = len(dimensions) - 1
        layers = []
        for i in range(self.layer_count):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            if i != self.layer_count - 1:
                layers.append(nn.ReLU())

        self.ffw = nn.Sequential(*layers)
        self.loss_fun = loss_fun
        self.optimizer = optimizer(self.parameters())

    def forward(self, x):
        return self.ffw(x)

    def train(self, mini_batches):
        self.ffw.train()
        loss = None
        for x, y in mini_batches:
            self.zero_grad()
            loss = self.loss_fun(self.__call__(x), y)
            loss.backward()
            self.optimizer.step()

        return loss


    def predict(self, x):
        self.ffw.eval()
        return self.__call__(x)


class QLearner:
    def __init__(self, env, model, epsilon_decay=0.998, epsilon_min=0.1, gamma=0.99):
        self.env = env
        self.model = model
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_min
        self.gamma = gamma

    def get_experience_replay_data(self, replay_memory, replay_size):
        # current states, actions, rewards, next states and 'done' flags
        experience_sample = random.sample(replay_memory, replay_size)

        # current states
        states = [es[0] for es in experience_sample]
        states = torch.tensor(states)
        q_vals = self.model.predict(states)

        # next states
        states_next = [es[3] for es in experience_sample]
        states_next = torch.tensor(states_next)
        q_vals_next = self.model.predict(states_next)

        for i in range(replay_size):
            a = experience_sample[i][1]
            r = experience_sample[i][2]

            # if done
            if experience_sample[i][4]:
                q_vals[i, a] = r
            else:
                q_vals[i, a] = float(r + self.gamma * max(q_vals_next[i]))

        return states, q_vals

    def run(self,
            num_episodes=2000,
            current_epsilon=1.0,
            replay_memory=2 ** 16,
            replay_sample_size=32,
            training_start_memory_size=64,
            most_recent_count=100,
            logging=True,
            train=True,
            render=False):

        logs = None
        if type(logging) is pd.DataFrame:
            logs = logging
            logging = True

        env, model = self.env, self.model

        if train:
            replay_memory = deque([], maxlen=replay_memory)
        recent_rewards = [0] * most_recent_count

        for episode_idx in range(num_episodes):

            total_reward = 0
            current_state = env.reset()
            done = False
            action_count = 0

            train_start_time = time()
            while not done:
                prev_state = current_state

                # epsilon greedy strategy to select the actions to take
                if current_epsilon > 0 and random.random() < current_epsilon:
                    act = env.action_space.sample()
                else:
                    current_state = torch.from_numpy(current_state)
                    act = int(torch.argmax(model.predict(current_state)))

                # interact with the environment
                current_state, reward, done, info = env.step(act)

                if render:
                    env.render();
                if train:
                    replay_memory.append((prev_state, act, reward, current_state, done))
                    if len(replay_memory) >= training_start_memory_size:
                        loss = float(self.model.train([self.get_experience_replay_data(replay_memory, replay_sample_size)]))
                    else:
                        loss = 0
                total_reward += reward
                action_count += 1
            curr_time_use = time() - train_start_time

            if current_epsilon > self.epsilon_end:
                current_epsilon *= self.epsilon_decay

            recent_rewards[episode_idx % most_recent_count] = total_reward

            most_rent_mean_reward = sum(recent_rewards) / most_recent_count

            log_entry = (
                episode_idx, total_reward, most_rent_mean_reward, loss, current_epsilon, action_count, curr_time_use)
            if logging:
                if logs is None:
                    logs = pd.DataFrame(columns=("Episode", "Total Reward", "Mean Reward", "Train Loss", "Epsilon", "Actions", "Training Time"))
                logs.loc[episode_idx] = log_entry
            print("Episode:{}, Total Reward: {}, Mean Reward: {:.2f}, Train Loss: {:.3f}, Epsilon: {:.5f}, Actions: {}, Training Time: {:.3f}".format(*log_entry))

        logs["Episode"] = logs["Episode"].astype(int)
        logs["Actions"] = logs["Actions"].astype(int)
        return most_rent_mean_reward, logs

    def train(self, episode_count=2500, epsilon_start=1.0, replay_memory=2 ** 16, replay_sample_size=32,
              training_start_memory_size=64, mean_reward_recency=100,
              logging=True, render=False):
        return self.run(num_episodes=episode_count,
                        current_epsilon=epsilon_start,
                        replay_memory=replay_memory,
                        replay_sample_size=replay_sample_size,
                        training_start_memory_size=max(replay_sample_size, training_start_memory_size),
                        most_recent_count=mean_reward_recency,
                        logging=logging, train=True, render=render)

    def test(self, start_episode_idx=0, episode_count=100, logging=True, continued_learning=True, replay_memory=2 ** 16,
             replay_sample_size=32, render=False):
        return self.run(start_episode_idx=start_episode_idx,
                        num_episodes=episode_count,
                        current_epsilon=0,
                        replay_memory=replay_memory,
                        replay_sample_size=replay_sample_size,
                        most_recent_count=episode_count,
                        logging=logging, train=continued_learning, render=render)


def train_lunar_lander(env, hidden_layer_dimensions=[128, 64],
                       training_episode_count=2000,
                       alpha=1e-4,
                       gamma=0.99,
                       epsilon_start=1.0,
                       epsilon_decay=0.998,
                       epsilon_min=0.0,
                       replay_memory_size=2 ** 16,
                       replay_sample_size=32,
                       training_start_memory_size=64,
                       most_recent_count=100):

    # set the dimensions of the model
    # the input size is the dimension of the observation, the output size is the dimension of the action space
    dimensions = [env.observation_space.shape[0]] + hidden_layer_dimensions + [env.action_space.n]

    ffw = FeedForwardNetwork(dimensions=dimensions,
                             loss_fun=nnF.mse_loss,
                             optimizer=lambda mode_paras: torch.optim.Adam(mode_paras, lr=alpha))

    lunar_lander = QLearner(env=env,
                            model=ffw,
                            epsilon_decay=epsilon_decay,
                            epsilon_min=epsilon_min,
                            gamma=gamma)

    mean_reward, logs = lunar_lander.train(episode_count=training_episode_count,
                                           epsilon_start=epsilon_start,
                                           replay_memory=replay_memory_size,
                                           replay_sample_size=replay_sample_size,
                                           training_start_memory_size=training_start_memory_size,
                                           mean_reward_recency=most_recent_count)

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    train_lunar_lander(env,
                       hidden_layer_dimensions=[128, 64],
                       training_episode_count=2000,
                       alpha=1e-4,
                       gamma=0.99,
                       epsilon_start=1.0,
                       epsilon_decay=0.998,
                       epsilon_min=0.0,
                       replay_memory_size=2 ** 16,
                       replay_sample_size=32,
                       training_start_memory_size=64,
                       most_recent_count=100)
