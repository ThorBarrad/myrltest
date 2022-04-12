import numpy as np
import pandas as pd


class Maze(object):
    def __init__(self):
        self.action_space = ["u", "d", 'l', 'r']
        # go from [1,1] to [8,8]
        self.env = [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ]
        self.locx = 0
        self.locy = 0

    def step(self, action):
        if action == "u":
            if self.locy > 0:
                self.locy -= 1
        elif action == "d":
            if self.locy < 3:
                self.locy += 1
        elif action == "l":
            if self.locx > 0:
                self.locx -= 1
        elif action == "r":
            if self.locx < 3:
                self.locx += 1

        if self.env[self.locy][self.locx] == 1:
            terminal = True
            reward = -1
        elif self.locx == 2 and self.locy == 2:
            terminal = True
            reward = 1
        else:
            terminal = False
            reward = 0
        return str([self.locy, self.locx]), reward, terminal

    def reset(self):
        self.locx = 0
        self.locy = 0
        return str([self.locy, self.locx])
