import numpy as np
from RL_brain import DDPG
from environment import Environment
import matplotlib.pyplot as plt

#####################  hyper parameters  ####################
EPISODES = 500
EP_STEPS = 2000
MEMORY_CAPACITY = 10000

############################### Training ######################################
# Define the env in gym
env = Environment(dog_v=4, sheep_v=1)
s_dim = 2
a_dim = 1
a_bound = 1
a_low_bound = 0
var = 3  # the controller of exploration which will decay during training process

ddpg = DDPG(a_dim, s_dim, a_bound)
for i in range(EPISODES):
    s = env.reset()
    ep_r = 0

    seita = []
    sheep_r = []

    for j in range(EP_STEPS):
        # add explorative noise to action
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), a_low_bound, a_bound)
        s_, r, done = env.step(a)
        # print(s, a, r, s_)
        ddpg.store_transition(s, a, r, s_)  # store the transition to memory

        if ddpg.pointer > MEMORY_CAPACITY and var > 0.01:
            var *= 0.9995  # decay the exploration controller factor
            ddpg.learn()

        s = s_
        ep_r += r

        seita.append(s[0])
        sheep_r.append(s[1])

        if j == EP_STEPS - 1 or done:
            print('Episode: ', i, ' Reward: %i' % (ep_r), 'Explore: %.2f' % var)
            break

    if i % 10 == 9:
        # var -= 0.1

        plt.figure(figsize=(10, 5))
        plt.plot(range(len(seita)), seita, label="seita")
        plt.plot(range(len(sheep_r)), sheep_r, color="r", label="sheep_r")
        plt.legend()
        plt.grid()
        plt.show()
