from env import Board
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

LR = 0.01
LR_C = 0.01
C = 2
TEMPERATURE = 1
ITERATION = 1000
MAX_DEPTH = 5
TOTAL_GAMES = 1024
BATCH_SIZE = 32
UPDATE_TIME = 16
# UPDATE_TIME = 1
INTERVAL = 8
# INTERVAL = 1

class TreeNode(object):
    def __init__(self, c):
        self.visit = 0
        self.value = 0
        self.action = -1
        self.parent = None
        self.child = []
        self.prob = 0
        self.c = c

    def __gt__(self, other):
        # if self.visit == 0:
        #     s_q = 0
        # else:
        #     s_q = self.value / self.visit
        # if other.visit == 0:
        #     o_q = 0
        # else:
        #     o_q = other.value / other.visit
        # s_u = self.c * self.prob * (self.parent.visit ** 0.5) / (1 + self.visit)
        # o_u = other.c * other.prob * (other.parent.visit ** 0.5) / (1 + other.visit)
        # return s_q + s_u > o_q + o_u

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
        childnode = TreeNode(C)
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
    def __init__(self, iteration, max_depth):
        self.node = TreeNode(C)
        self.iteration = iteration
        self.max_depth = max_depth

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
            step_count = 0
            while not state.end and step_count < self.max_depth:
                prob_list = network.predict(state.get_board())
                new_list = prob_list * state.valid_moves
                sum = new_list.sum()
                action_list = [prob / sum for prob in new_list]
                state.step(np.random.choice(len(action_list), p=action_list))
                step_count += 1
            if step_count == self.max_depth:
                winner = network_c.predict(state.get_board())[0]
                self.node = self.node.backup(-winner * turn)
            else:
                self.node = self.node.backup(-state.winner * turn)

        ans = np.zeros(len(state_.valid_moves))
        sums = 0
        for i in self.node.child:
            sums += i.visit ** temperature
        for j in self.node.child:
            ans[j.action] = (j.visit ** temperature) / sums
        return ans


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=(3, 3))
        self.fc1 = nn.Linear(160, 120)
        self.fc2 = nn.Linear(120, 81)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x):
        x = F.relu(self.maxpool2d(self.conv1(x)))
        x = x.view(-1, 160)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x)

    def predict(self, state):
        return self.forward(torch.Tensor(state).unsqueeze(0)).detach().numpy()[0]


class NetC(nn.Module):
    def __init__(self):
        super(NetC, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=(3, 3))
        self.fc1 = nn.Linear(160, 120)
        self.fc2 = nn.Linear(120, 1)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x):
        x = F.relu(self.maxpool2d(self.conv1(x)))
        x = x.view(-1, 160)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.clamp(x, -1, 1)

    def predict(self, state):
        return self.forward(torch.Tensor(state).unsqueeze(0)).detach().numpy()[0]


network = Net()
# network = torch.load("netG.pt")
loss_func = nn.MSELoss()
optimizer = optim.SGD(network.parameters(), lr=LR)

network_c = NetC()
# network_c = torch.load("netC.pt")
loss_func_c = nn.MSELoss()
optimizer_c = optim.SGD(network_c.parameters(), lr=LR_C)

state_list = []
action_list = []
winner_list = []
for game_count in range(1, TOTAL_GAMES + 1):
    state_temp = []
    action_temp = []
    winner_temp = []
    print("game", game_count, "started")
    chessboard = Board()
    while not chessboard.end:
        chessboard.render()
        tree = TreeSearch(ITERATION, MAX_DEPTH)
        action_prob = tree.iterate(chessboard, temperature=TEMPERATURE)

        for rot in range(4):
            for mir in range(2):
                state_temp.append(chessboard.get_board())
                action_temp.append(action_prob)

                chessboard.mirror()
                temp = action_prob.reshape((chessboard.row, chessboard.col))
                temp = np.flip(temp, axis=1)
                action_prob = temp.reshape(chessboard.row * chessboard.col)

            chessboard.rotate()
            temp = action_prob.reshape((chessboard.row, chessboard.col))
            temp = np.rot90(temp, 1)
            action_prob = temp.reshape(chessboard.row * chessboard.col)

        action = np.random.choice(len(action_prob), p=action_prob)
        chessboard.step(action)
        print("stepped at: (", action // chessboard.row, ",", action % chessboard.row, ")")

    for i in range(len(state_temp)):
        winner_temp.append(np.array([chessboard.winner]))

    state_list.extend(state_temp)
    action_list.extend(action_temp)
    winner_list.extend(winner_temp)

    chessboard.render()

    if game_count % INTERVAL == 0:
        # update network
        for t in range(UPDATE_TIME):
            indexs = np.random.choice(np.arange(len(state_list)), size=BATCH_SIZE, replace=False)
            x_train = torch.Tensor(np.array(state_list)[indexs])
            y_train = torch.Tensor(np.array(action_list)[indexs])
            y_train_c = torch.Tensor(np.array(winner_list)[indexs])

            y_pred = network.forward(x_train)
            loss = loss_func(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred_c = network_c.forward(x_train)
            loss_c = loss_func_c(y_pred_c, y_train_c)
            optimizer_c.zero_grad()
            loss_c.backward()
            optimizer_c.step()

        state_list = []
        action_list = []
        winner_list = []

        torch.save(network, "netG.pt")
        torch.save(network_c, "netC.pt")

print("training complete!")

# for game in range(100):
#     print("game:", game, "started")
#     chessboard = Board()
#     state_list = []
#     action_list = []
#     winner_list = []
#
#     while not chessboard.end:
#         chessboard.render()
#         tree = TreeSearch(ITERATION, MAX_DEPTH)
#
#         action_prob = tree.iterate(chessboard, temperature=TEMPERATURE)
#
#         state_list.append(chessboard.get_board())
#         action_list.append(action_prob)
#
#         action = np.random.choice(len(action_prob), p=action_prob)
#         chessboard.step(action)
#         print("stepped at: (", action // chessboard.row, ",", action % chessboard.row, ")")
#
#     for i in range(len(state_list)):
#         winner_list.append(np.array([chessboard.winner]))
#     chessboard.render()
#
#     for i in range(len(state_list)):
#         x_train = torch.Tensor(state_list[i]).unsqueeze(0)
#         y_train = torch.Tensor(action_list[i])
#         y_pred = network.forward(x_train)
#         loss = loss_func(y_pred, y_train)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         x_train_c = torch.Tensor(state_list[i]).unsqueeze(0)
#         y_train_c = torch.Tensor(winner_list[i])
#         y_pred_c = network_c.forward(x_train_c)
#         loss_c = loss_func_c(y_pred_c, y_train_c)
#         optimizer_c.zero_grad()
#         loss_c.backward()
#         optimizer_c.step()
#
#     torch.save(network, "netG.pt")
#     torch.save(network_c, "netC.pt")
