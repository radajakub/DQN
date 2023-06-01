import numpy as np
import torch as th
from torch import nn
from torch import optim
import random

from collections import deque

np.random.seed(1)


class DeepQLearning:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate,
            discount_factor,
            e_greedy,
            replace_target_iter,
            memory_size,
            batch_size
    ):
        # store paremeters into object attributes
        self.action_count = n_actions
        self.feature_count = n_features
        self.gamma = discount_factor
        self.epsilon = e_greedy
        self.batch_size = batch_size

        self.replace_target_counter = 0
        self.replace_target_iter = replace_target_iter

        self.hidden_size = 100

        # initialize experience replay buffer
        # deque -> with maxlen specified, when inserted element makes the deque larger than the maxlen, the corresponding number of elements are removed from the opposite side
        self.experience = deque([], maxlen=memory_size)

        # construct two networks
        self.Q_target = self.construct_network()
        self.Q_estim = self.construct_network()

        self.optimizer = optim.Adam(
            self.Q_estim.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    # builds a torch network with given architecture
    def construct_network(self):
        return nn.Sequential(
            # one input layer taking a state as an input
            nn.Linear(self.feature_count, self.hidden_size),
            nn.ReLU(),
            # # one hidden layer
            # nn.Linear(self.hidden_size, self.hidden_size),
            # nn.ReLU(),
            # one output layer which returns Q values of all actions for given input state
            nn.Linear(self.hidden_size, self.action_count),
        ).float()

    def store_transition(self, s, a, r, next_s):
        # save tuple into experience buffer
        # deque takes care about forgetting old experiences
        self.experience.append((s, a, r, next_s))

    def choose_action(self, state):
        # take random action with probability epsilon
        if random.random() < self.epsilon:
            return random.randrange(self.action_count)

        state = th.from_numpy(state).float()
        # otherwise take best action according to Q_estim i.e. argmax_a Q_estim(s, a)
        with th.no_grad():
            a_vals = self.Q_estim(state)
        return th.argmax(a_vals).item()

    def learn(self):
        self.optimizer.zero_grad()

        # draw batch from experience replay buffer
        batch = random.sample(self.experience, self.batch_size)

        # split tuples in the batch by element types (s, a, r, s')
        transposed = list(zip(*batch))
        states = th.tensor(np.array(transposed[0])).float()
        actions = th.tensor(transposed[1])
        rewards = th.tensor(transposed[2]).float()
        next_states = th.tensor(np.array(transposed[3])).float()

        # compute targets from the frozen neural network
        with th.no_grad():
            # target = R_t+1 + gamma * max_a q~(s_t+1, a, w-)
            # add detach because we dont want to compute gradient w.r.t. to targets
            targets = rewards + self.gamma * \
                self.Q_target(next_states).detach().max(1)[0]

        # compute predictions q^(s_t, a_t, w_t)
        estimates = self.Q_estim(states)[range(len(actions)), actions]

        # select MSE as loss and compute its gradient
        loss = self.criterion(targets, estimates) / 2
        loss.backward()

        # make a step in opitmization
        self.optimizer.step()

        # if the counter reached replace_target_iter, freeze current weights
        self.replace_target_counter += 1
        if self.replace_target_counter == self.replace_target_iter:
            self.replace_target_counter = 0
            self.Q_target.load_state_dict(self.Q_estim.state_dict())
