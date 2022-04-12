from RL_brain import DDQN
from copy import deepcopy
import numpy as np
import gym


def update():
    totalstep = 0
    state = env.reset()
    while True:

        env.render()

        # print(state)

        action = QLearning.choose_action(state)

        # print(state, action)

        state_, reward, is_end, info = env.step([action])

        QLearning.store_transaction(state, action, reward, state_, is_end)

        state = state_

        if totalstep > QLearning.pool_size:
            QLearning.learn()

        totalstep = totalstep + 1

        if totalstep % 200 == 0:
            print("totalstep", totalstep, "target network updated")
            QLearning.network_target = deepcopy(QLearning.network_now)

            # if QLearning.epsilon < 0.95:
            #     QLearning.epsilon = QLearning.epsilon + 0.01

        if totalstep > 20000:
            break


if __name__ == "__main__":
    env = gym.make("Pendulum-v0")
    env = env.unwrapped
    QLearning = DDQN(action_space=[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
    update()
