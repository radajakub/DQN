import torch as th
from torch import optim
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
from collections import deque


class Reinforce:
    def __init__(self, n_actions, n_features, learning_rate, discount_factor, eps):
        self.action_count = n_actions
        self.feature_count = n_features
        self.gamma = discount_factor
        self.eps = eps

        # build network
        self.policy = self.construct_network()
        # create optimizer which will handle updates of the neural network
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # variables from the mainfile
        self.rewards = []
        self.action_log_probs = []

    def construct_network(self):
        # create network
        return nn.Sequential(
            nn.Linear(self.feature_count, 100),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(100, self.action_count),
            # it is a policy approximuation, thus should return probabilities
            nn.Softmax(dim=-1)
        ).float()

    def choose_action(self, state):
        # obtain probabilities from neural network, i.e. current estimation of policy pi
        state_tensor = th.from_numpy(state).unsqueeze(0).float()
        a_probs = self.policy(state_tensor)
        #  and create a categorical distribution over them
        prob_dist = Categorical(probs=a_probs)
        # obtain sample form the categorical distribution -> action index from [0, ..., k-1]
        action = prob_dist.sample()
        # save log probability tensor of the action
        self.action_log_probs.append(prob_dist.log_prob(action))
        # return the selected action
        return action.item()

    def learn(self):
        # generate returns for every step
        G = 0
        returns = deque([])
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.appendleft(G)
        # convert to tensor
        returns = th.tensor(returns)

        loss = []
        for action_log_prob, ret in zip(self.action_log_probs, returns):
            # add -G * gamma^t * log_pi (gradient will be computed later, G is multiplicative constant, will hold through differentiation)
            loss.append(-ret * action_log_prob)

        self.optimizer.zero_grad()
        loss = th.cat(loss).sum()
        # compute the gradient
        loss.backward()
        self.optimizer.step()

        # clear memory after learning the episode
        self.rewards.clear()
        self.action_log_probs.clear()
