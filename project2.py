import torch.nn as nn
import torch
import gym
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

    def train(self, states, qvalues):
        self.ffw.train()
        self.optimizer.zero_grad()
        loss = self.loss_fun(self.__call__(states), qvalues)
        loss.backward()
        self.optimizer.step()
        return loss


    def predict(self, x):
        self.ffw.eval()
        return self.__call__(x)


class QLearner:
    def __init__(self, env, model, epsilon_decay, epsilon_min, gamma):
        self.env = env
        self.model = model
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_min
        self.gamma = gamma

    def get_experience_replay_data(self, replay_memory, replay_size):
        # current states, actions, rewards, next states and done
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

    def train(self,
              num_episodes,
              current_epsilon,
              alpha,
              gamma,
              replay_memory,
              replay_sample_size,
              training_start_memory_size,
              most_recent_count):


        # initial set up
        logs = None
        env, model = self.env, self.model
        replay_memory = deque([], maxlen=replay_memory)
        recent_rewards = [0] * most_recent_count

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
                    act = int(torch.argmax(model.predict(torch.from_numpy(current_state))))

                # interact with the environment
                next_state, reward, done, info = env.step(act)

                # add to the replay memory
                replay_memory.append((current_state, act, reward, next_state, done))
                if len(replay_memory) >= training_start_memory_size:
                    states, qvalues = self.get_experience_replay_data(replay_memory, replay_sample_size)
                    loss = float(self.model.train(states, qvalues))

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

            log_entry = (episode_index, total_reward, most_rent_mean_reward, loss, current_epsilon)
            if logs is None:
                logs = pd.DataFrame(columns=("Episode", "Total Reward", "Mean Reward", "Train Loss", "Epsilon"))
            logs.loc[episode_index] = log_entry
            print("Episode:{}, Current Reward: {}, Mean Reward: {:.3f}, Train Loss: {:.3f}, Epsilon: {:.5f}".format(*log_entry))

        # save the log to csv
        logs.to_csv("result_gamma{}_alpha{}.csv".format(gamma, alpha), index=False)

        # save the model
        torch.save(model.state_dict(), "result_gamma{}_alpha{}.model".format(gamma, alpha))

        return

    def test(self, num_episodes, dimensions, gamma, alpha):
        return
        model = FeedForwardNetwork(dimensions=dimensions,
                                   loss_fun=nnF.mse_loss,
                                   optimizer=lambda parameters: torch.optim.Adam(parameters, lr=alpha))
        model.load_state_dict(torch.load("result_gamma{}_alpha{}.model").format(gamma, alpha))
        model.eval()


if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    hidden_layer_dimensions = [128, 64]
    training_episode_count = 1000
    alpha = 1e-4
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_decay = 0.998
    epsilon_min = 0.0
    replay_memory_size = 65536
    replay_sample_size = 32
    training_start_memory_size = 64
    most_recent_count = 100

    # set the dimensions of the model
    # the input size is the dimension of the observation, the output size is the dimension of the action space
    dimensions = [env.observation_space.shape[0]] + hidden_layer_dimensions + [env.action_space.n]

    ffw = FeedForwardNetwork(dimensions=dimensions,
                             loss_fun=nnF.mse_loss,
                             optimizer=lambda parameters: torch.optim.Adam(parameters, lr=alpha))



    lunar_lander = QLearner(env=env,
                            model=ffw,
                            epsilon_decay=epsilon_decay,
                            epsilon_min=epsilon_min,
                            gamma=gamma)


    lunar_lander.train(num_episodes=training_episode_count,
                       current_epsilon=epsilon_start,
                       alpha = alpha,
                       gamma = gamma,
                       replay_memory=replay_memory_size,
                       replay_sample_size=replay_sample_size,
                       training_start_memory_size=max(replay_sample_size, training_start_memory_size),
                       most_recent_count=most_recent_count)
