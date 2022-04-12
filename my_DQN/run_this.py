from maze_env import Maze
from RL_brain import DDQN
from copy import deepcopy
import numpy as np


def update():
    count = 0
    for i in range(500):
        state = env.reset()
        stepcount = 0
        while True:

            action = QLearning.choose_action(state)
            # print(state,action)

            state_, reward, is_end = env.step(action)
            QLearning.store_transaction(state, action, reward, state_, is_end)
            state = state_

            QLearning.learn()
            QLearning.learn_positive()

            count = count + 1
            if count == 100:
                QLearning.network_target = deepcopy(QLearning.network_now)
                count = 0

            stepcount = stepcount + 1
            if is_end:
                print("iteration", i, "complete, total step is", stepcount, "reward is", reward)
                break


def run():
    state = env.reset()
    QLearning.epsilon = 1
    is_end = False
    while is_end is not True:
        action = QLearning.choose_action(state)
        state_, reward, is_end = env.step(action)
        print(state, action, state_)
        state = state_


if __name__ == "__main__":
    env = Maze()
    QLearning = DDQN(action_space=env.action_space)
    update()
    run()
