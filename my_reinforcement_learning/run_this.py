from maze_env import Maze
from RL_brain import QLearningTable, SarsaTable
import time
import numpy as np


def update_q():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        totalstep = 0

        while True:
            # RL choose action based on observation
            action = QLearning.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            QLearning.learn(observation, action, reward, observation_, done)

            # swap observation
            observation = observation_

            totalstep = totalstep + 1

            # break while loop when end of this episode
            if done:
                print("iteration", episode, "done, total step is", totalstep)
                break

    # end of game
    print('game over')


def update_s():
    for episode in range(500):
        # initial observation
        observation = env.reset()

        # RL choose action based on observation
        action = Sarsa.choose_action(observation)

        totalstep = 0

        while True:
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on next observation
            action_ = Sarsa.choose_action(observation_)

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            Sarsa.learn(observation, action, reward, observation_, action_, done)

            # swap observation and action
            observation = observation_
            action = action_

            totalstep = totalstep + 1

            # break while loop when end of this episode
            if done:
                print("iteration", episode, "done, total step is", totalstep)
                break

    # end of game
    print('game over')


def run_q():
    print()
    observation = env.reset()
    done = False
    while done is not True:
        print(observation)
        state_action = QLearning.q_table.loc[observation, :]
        action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        print(action)
        observation_, reward, done = env.step(action)
        observation = observation_


def run_s():
    print()
    observation = env.reset()
    done = False
    while done is not True:
        print(observation)
        state_action = Sarsa.q_table.loc[observation, :]
        action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        print(action)
        observation_, reward, done = env.step(action)
        observation = observation_


if __name__ == "__main__":
    env = Maze()

    QLearning = QLearningTable(actions=env.action_space)
    update_q()
    print(QLearning.q_table)
    run_q()

    Sarsa = SarsaTable(actions=env.action_space)
    update_s()
    print(Sarsa.q_table)
    run_s()
