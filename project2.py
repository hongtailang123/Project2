import torch.nn as nn
import torch
import gym
import pandas as pd
from collections import deque
import random
import torch.nn.functional as nnF

class LunarLander:
    def __init__(self, env, epsilon_start, epsilon_decay, epsilon_min, hidden_layer_dimensions, gamma, alpha, replay_memory_size, replay_sample_size, training_start_memory_size):
        self.env = env
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_min
        self.hidden_layer_dimensions = hidden_layer_dimensions
        self.gamma = gamma
        self.alpha = alpha
        self.replay_memory_size = replay_memory_size
        self.replay_sample_size = replay_sample_size
        self.training_start_memory_size = training_start_memory_size


    def get_experience_replay_data(self, replay_memory, replay_size, ffw):
        # current states, actions, rewards, next states and done
        experience_sample = random.sample(replay_memory, replay_size)

        # current states
        states = [es[0] for es in experience_sample]
        states = torch.tensor(states)
        q_vals = ffw(states)

        # next states
        states_next = [es[3] for es in experience_sample]
        states_next = torch.tensor(states_next)
        q_vals_next = ffw(states_next)

        for i in range(replay_size):
            a = experience_sample[i][1]
            r = experience_sample[i][2]

            # if done
            if experience_sample[i][4]:
                q_vals[i, a] = r
            else:
                q_vals[i, a] = float(r + self.gamma * max(q_vals_next[i]))

        return states, q_vals

    def train(self, num_episodes):


        # initial set up
        logs = None
        env = self.env
        replay_memory = deque([], maxlen=self.replay_memory_size)
        most_recent_count = 100
        recent_rewards = [0] * most_recent_count
        current_epsilon = self.epsilon_start

        # set the dimensions of the model
        # the input size is the dimension of the observation, the output size is the dimension of the action space
        dimensions = [env.observation_space.shape[0]] + hidden_layer_dimensions + [env.action_space.n]

        # set up the neural network
        layer_count = len(dimensions) - 1
        layers = []
        for i in range(layer_count):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            if i != layer_count - 1:
                layers.append(nn.ReLU())

        ffw = nn.Sequential(*layers)
        loss_fun = nnF.mse_loss
        optimizer = torch.optim.Adam(ffw.parameters(), lr=self.alpha)


        for episode_index in range(num_episodes):

            total_reward = 0
            current_state = env.reset()
            done = False
            action_count = 0

            while not done:

                # epsilon greedy strategy to select the actions to take
                if current_epsilon > 0 and random.random() < current_epsilon:
                    act = env.action_space.sample()
                else:
                    act = int(torch.argmax(ffw(torch.from_numpy(current_state))))

                # interact with the environment
                next_state, reward, done, info = env.step(act)

                # add to the replay memory
                replay_memory.append((current_state, act, reward, next_state, done))
                if len(replay_memory) >= self.training_start_memory_size:
                    states, qvalues = self.get_experience_replay_data(replay_memory, self.replay_sample_size, ffw)
                    optimizer.zero_grad()
                    loss = loss_fun(ffw(states), qvalues)
                    loss.backward()
                    optimizer.step()

                # update state, reward, action count
                current_state = next_state
                total_reward += reward
                action_count += 1

            # update current_epsilon
            if current_epsilon > self.epsilon_end:
                current_epsilon *= self.epsilon_decay

            # record the most recent episode reward
            recent_rewards[episode_index % most_recent_count] = total_reward
            if episode_index >= 100:
                most_rent_mean_reward = sum(recent_rewards) / most_recent_count
            else:
                most_rent_mean_reward = sum(recent_rewards) / (episode_index + 1)

            log_entry = (episode_index, total_reward, most_rent_mean_reward, float(loss), current_epsilon)
            if logs is None:
                logs = pd.DataFrame(columns=("Episode", "Total Reward", "Mean Reward", "Train Loss", "Epsilon"))
            logs.loc[episode_index] = log_entry
            print("Episode:{}, Current Reward: {:.3f}, Mean Reward: {:.3f}, Train Loss: {:.3f}, Epsilon: {:.5f}".format(*log_entry))

        # save the log to csv
        logs.to_csv("result_gamma{}_alpha{}_initialEpsilon{}_decay{}_hidden1{}_hidden2{}.csv".
                    format(self.gamma, self.alpha, self.initial_epsilon, self.epsilon_decay, self.hidden_layer_dimensions[0], self.hidden_layer_dimensions[1]), index=False)

        # save the model
        torch.save(ffw.state_dict(), "result_gamma{}_alpha{}.model".format(self.gamma, self.alpha))

        return

    def test(self, num_episodes, hidden_layer_dimensions, model_name):
        dimensions = [env.observation_space.shape[0]] + hidden_layer_dimensions + [env.action_space.n]

        # set up the neural network
        layer_count = len(dimensions) - 1
        layers = []
        for i in range(layer_count):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            if i != layer_count - 1:
                layers.append(nn.ReLU())

        ffw = nn.Sequential(*layers)
        ffw.load_state_dict(torch.load(model_name))

        for episode_index in range(num_episodes):
            current_state = env.reset()
            done = False

            while not done:
                act = int(torch.argmax(ffw(torch.from_numpy(current_state))))

                # interact with the environment
                next_state, reward, done, info = env.step(act)
                env.render()
                current_state = next_state



if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    hidden_layer_dimensions = [128, 64]
    training_episode_count = 1200
    alpha = 1e-4
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_decay = 0.998
    epsilon_min = 0.01
    replay_memory_size = 65536
    replay_sample_size = 32
    training_start_memory_size = 64
    most_recent_count = 100

    lunar_lander = LunarLander(env=env,
                            epsilon_start=epsilon_start,
                            epsilon_decay=epsilon_decay,
                            epsilon_min=epsilon_min,
                            hidden_layer_dimensions=hidden_layer_dimensions,
                            gamma=gamma,
                            alpha=alpha,
                            replay_memory_size=replay_memory_size,
                            replay_sample_size=replay_sample_size,
                            training_start_memory_size=training_start_memory_size)

    # lunar_lander.test(num_episodes=20, hidden_layer_dimensions=hidden_layer_dimensions, model_name="result_gamma0.99_alpha0.0001.model")


    lunar_lander.train(num_episodes=training_episode_count)
