import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from copy import deepcopy
import random


class DDQN(object):
    def __init__(self, action_space, reward_decay=0.9, e_greedy=0.9, batch_size=32, pool_size=3000):
        self.actions = action_space  # a list
        self.pool_size = pool_size
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.network_now = MLPRegressor()
        # self.network_now.set_params()
        self.network_now.fit([[0, 0, 0, 0]], [0])
        self.network_target = deepcopy(self.network_now)
        self.exp_pool = []
        self.batch_size = batch_size

    def choose_action(self, state):
        state_action = []

        for i in self.actions:
            # state.append(i)

            # print(state, i)

            sa = np.insert(state, 3, values=i)
            state_action.append(self.network_now.predict([sa]))
            # state.pop()

        if np.random.rand() < self.epsilon:
            # choose best action
            # some actions may have the same value, randomly choose on in these actions
            action = self.actions[state_action.index(max(state_action))]

        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def store_transaction(self, state, action, reward, state_, is_end):

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
                for j in self.actions:
                    # state_.append(j)
                    sa_ = np.insert(state_, 3, values=j)
                    state_action.append(self.network_now.predict([sa_]))
                    # state_.pop()

                action_ = self.actions[state_action.index(max(state_action))]

                # state_.append(action_)
                sa_ = np.insert(state_, 3, values=action_)
                q = self.network_target.predict([sa_])
                # state_.pop()
                q_target = reward + self.gamma * q

            else:
                q_target = [reward]

            # state.append(action)
            sa = np.insert(state, 3, values=action)
            self.network_now.partial_fit([sa], q_target)
            # state.pop()
