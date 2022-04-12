from math import pi, sin, cos
import matplotlib.pyplot as plt
import random


class Environment(object):
    def __init__(self, dog_v, sheep_v, delta_t=0.01):
        self.dog_v = dog_v
        self.sheep_v = sheep_v
        self.seita = 1  # 1 * pi, if seita < 0, dog wins
        self.sheep_r = 0.01  # if sheep_r > 1, sheep wins
        self.delta_t = delta_t  # time step

    def reset(self):
        self.seita = 1
        self.sheep_r = 0.01
        return [self.seita, self.sheep_r]

    def step(self, action):  # action should be (in 0-1), use action * pi / 2

        delta_sheep_r = self.sheep_r

        self.sheep_r = self.sheep_r + self.sheep_v * cos(action * pi / 2) * self.delta_t

        delta_sheep_r = self.sheep_r - delta_sheep_r

        delta_seita = self.seita

        self.seita = self.seita - (self.dog_v * self.delta_t) / pi

        self.seita = self.seita + (self.sheep_v * sin(action * pi / 2) * self.delta_t) / (self.sheep_r * pi)

        self.seita = min(self.seita, 2 - self.seita)

        delta_seita = self.seita - delta_seita

        # print(delta_seita, delta_sheep_r)

        reward = delta_seita + delta_sheep_r

        if self.sheep_r > 1:
            # reward = 1
            is_end = True
        elif self.seita < 0:
            # reward = -1
            is_end = True
        else:
            # reward = 0
            is_end = False
        return [self.seita, self.sheep_r], reward, is_end


def update():
    for t in range(1):
        s = env.reset()
        seita = []
        sheep_r = []
        while True:
            a = random.random()  # (0-1) -> (0,pi/2)
            s_, r, done = env.step(a)
            print(s, a, r, s_, done)
            s = s_
            seita.append(s[0])
            sheep_r.append(s[1])
            if done:
                break
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(seita)), seita, label="seita")
        plt.plot(range(len(sheep_r)), sheep_r, color="r", label="sheep_r")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == '__main__':
    env = Environment(dog_v=4, sheep_v=1)
    update()
