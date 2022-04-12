import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from copy import deepcopy
import random


class DDQN(object):
    def __init__(self, action_space, reward_decay=0.9, e_greedy=0.9, batch_size=8, pool_size=200):
        self.actions = action_space  # a list
        self.pool_size = pool_size
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.network_now = MLPRegressor()
        # self.network_now.fit([[3, 2, 0], [2, 3, 2]], [1, 1])
        # self.network_now.fit([[0, 0, 0]], [0])
        self.network_target = deepcopy(self.network_now)
        self.exp_pool = []
        self.positive_pool = [
            {"state": [3, 2], "action": "u", "reward": 1, "state_": [2, 2], "is_end": True},
            {"state": [2, 3], "action": "l", "reward": 1, "state_": [2, 2], "is_end": True}
        ]
        self.batch_size = batch_size

    def choose_action(self, state):
        state_action = []
        for i in range(len(self.actions)):
            state.append(i)
            state_action.append(self.network_now.predict([state]))
            state.pop()
        if np.random.rand() < self.epsilon:
            # choose best action
            # some actions may have the same value, randomly choose on in these actions
            action = self.actions[state_action.index(max(state_action))]
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def store_transaction(self, state, action, reward, state_, is_end):
        # item = [state, action, reward, state_, is_end]
        item = {
            "state": state,
            "action": action,
            "reward": reward,
            "state_": state_,
            "is_end": is_end
        }
        self.exp_pool.append(item)
        if len(self.exp_pool) > self.pool_size:
            self.exp_pool.pop(0)

    def learn(self):
        for i in range(self.batch_size):
            item = random.choice(self.exp_pool)
            state = item['state']
            action = item['action']
            reward = item['reward']
            state_ = item['state_']
            is_end = item['is_end']
            if is_end is not True:

                state_action = []
                for j in range(len(self.actions)):
                    state_.append(j)
                    state_action.append(self.network_now.predict([state_]))
                    state_.pop()
                action_ = self.actions[state_action.index(max(state_action))]
                state_.append(self.actions.index(action_))
                q = self.network_target.predict([state_])
                state_.pop()
                q_target = reward + self.gamma * q
            else:
                q_target = [reward]
            state.append(self.actions.index(action))
            # print([state],q_target,is_end)
            self.network_now.partial_fit([state], q_target)
            state.pop()

    def learn_positive(self):
        item = random.choice(self.positive_pool)
        state = item['state']
        action = item['action']
        reward = item['reward']
        state_ = item['state_']
        is_end = item['is_end']
        if is_end is not True:

            state_action = []
            for j in range(len(self.actions)):
                state_.append(j)
                state_action.append(self.network_now.predict([state_]))
                state_.pop()
            action_ = self.actions[state_action.index(max(state_action))]
            state_.append(self.actions.index(action_))
            q = self.network_target.predict([state_])
            state_.pop()
            q_target = reward + self.gamma * q
        else:
            q_target = [reward]
        state.append(self.actions.index(action))
        # print([state],q_target,is_end)
        self.network_now.partial_fit([state], q_target)
        state.pop()
