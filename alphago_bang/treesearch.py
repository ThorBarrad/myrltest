from env import Board
from copy import deepcopy
import numpy as np


class TreeNode(object):
    def __init__(self):
        self.visit = 0
        self.value = 0
        self.action = -1
        self.parent = None
        self.child = []
        self.prob = 0
        self.c = 2

    def __gt__(self, other):
        if other.visit == 0 and self.visit == 0:
            return self.prob > other.prob
        elif other.visit == 0 and self.visit != 0:
            return False
        elif other.visit != 0 and self.visit == 0:
            return True
        else:
            return (self.value / self.visit) + self.c * self.prob * (self.parent.visit ** 0.5) / (1 + self.visit) > \
                   (other.value / other.visit) + other.c * other.prob * (other.parent.visit ** 0.5) / (1 + other.visit)

    def search(self):
        return max(self.child)

    def expand(self, prob_list):
        for action in prob_list:
            self.add_child(action[1], action[0])
        return self.search()

    def add_child(self, action, prob):
        childnode = TreeNode()
        childnode.action = action
        childnode.prob = prob
        childnode.parent = self
        self.child.append(childnode)

    def backup(self, value):
        self.visit += 1
        self.value += value
        if self.parent is not None:
            return self.parent.backup(-value)
        else:
            return self


class TreeSearch(object):
    def __init__(self):
        self.node = TreeNode()
        self.iteration = 1000

    def iterate(self, state_, temperature=1.0):
        for i in range(self.iteration):
            if i % 100 == 0:
                print("iteration:", i, "started")
            state = deepcopy(state_)
            while not state.end and len(self.node.child) > 0:
                self.node = self.node.search()
                state.step(self.node.action)
            if state.end:
                self.node = self.node.backup(1)
                continue
            if self.node.visit > 0:
                prob_list = network.predict(
                    state.get_board())  # prob_list=[0.1,0.1,0.3,0.4,0.1], while valid_moves=[0,1,1,0,1]
                new_list = prob_list * state.valid_moves
                sum = new_list.sum()
                action_list = [[new_list[i] / sum, i] for i in range(len(new_list)) if
                               new_list[i] > 0]  # action_list=[[0.2, 1],[0.6 ,1],[0.2, 1]]
                self.node = self.node.expand(action_list)
                state.step(self.node.action)
            turn = state.turn
            while not state.end:
                prob_list = network.predict(state.get_board())
                new_list = prob_list * state.valid_moves
                sum = new_list.sum()
                action_list = [prob / sum for prob in new_list]
                state.step(np.random.choice(len(action_list), p=action_list))
            if state.winner == 0:
                self.node = self.node.backup(0)
            elif state.winner == turn:
                self.node = self.node.backup(-1)
            else:
                self.node = self.node.backup(1)

        ans = np.zeros(len(state_.valid_moves))
        sums = 0
        for i in self.node.child:
            sums += i.visit ** temperature
        for j in self.node.child:
            ans[j.action] = (j.visit ** temperature) / sums
        return ans


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=(3, 3))
        self.fc1 = nn.Linear(160, 120)
        self.fc2 = nn.Linear(120, 81)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x):
        x = F.relu(self.maxpool2d(self.conv1(x)))
        x = x.view(160)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x)

    def predict(self, state):
        return self.forward(torch.Tensor(state).unsqueeze(0)).detach().numpy()


# network = Net()
network = torch.load("netG.pt")
loss_func = nn.MSELoss()
optimizer = optim.SGD(network.parameters(), lr=0.01)

for game in range(100):
    print("game:", game, "started")
    chessboard = Board()
    state_list = []
    action_list = []

    while not chessboard.end:
        chessboard.render()
        tree = TreeSearch()
        action_prob = tree.iterate(chessboard, temperature=0.9)
        state_list.append(chessboard.get_board())
        action_list.append(action_prob)
        chessboard.step(np.random.choice(len(action_prob), p=action_prob))
    chessboard.render()

    for i in range(len(state_list)):
        x_train = torch.Tensor(state_list[i]).unsqueeze(0)
        y_train = torch.Tensor(action_list[i])
        y_pred = network.forward(x_train)
        loss = loss_func(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(network, "netG.pt")
