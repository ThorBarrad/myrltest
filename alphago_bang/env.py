import numpy as np


class Board(object):
    def __init__(self, row=9, col=9, max_n=5):
        self.row = row
        self.col = col
        self.max_n = max_n
        self.turn = 1  # 1 for black, -1 for white
        self.winner = 0  # 1 for black, -1 for white, 0 for tie
        self.board = np.zeros((self.row, self.col))
        self.end = False
        self.valid_moves = np.ones(self.row * self.col)

    def render(self):
        print("      0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8")
        for r in range(self.row):
            print("    ----|---|---|---|---|---|---|---|---")
            print("%2d" % r, end=" => ")
            for c in range(self.col):
                if self.board[r][c] == 1:
                    print("B", end=" | ")
                elif self.board[r][c] == -1:
                    print("W", end=" | ")
                else:
                    print(".", end=" | ")
            print()
        # print("turn: B" if self.turn == 1 else "turn: W")
        # print("end:", self.end)
        # print("winner: B" if self.winner == 1 else "winner: W" if self.winner == -1 else "now tie")
        # print("valid moves:", self.valid_moves)

    def get_board(self):
        res_board = np.zeros((3, self.row, self.col))
        for r in range(self.row):
            for c in range(self.col):
                if self.board[r][c] == 1:
                    res_board[0][r][c] = 1
                elif self.board[r][c] == -1:
                    res_board[1][r][c] = 1
                if self.turn == 1:
                    res_board[2] = np.ones((self.row, self.col))
        return res_board

    def rotate(self):
        self.board = np.rot90(self.board, 1)
        temp = self.valid_moves.reshape((self.row, self.col))
        temp = np.rot90(temp, 1)
        self.valid_moves = temp.reshape(self.row * self.col)

    def mirror(self):
        self.board = np.flip(self.board, axis=1)
        temp = self.valid_moves.reshape((self.row, self.col))
        temp = np.flip(temp, axis=1)
        self.valid_moves = temp.reshape(self.row * self.col)

    def change_player(self):
        if self.turn == 1:
            return -1
        else:
            return 1

    def step(self, action):
        x = action // self.row
        y = action % self.row
        self.board[x][y] = self.turn
        self.valid_moves[action] = 0
        if self.valid_moves.sum() == 0:
            self.end = True
        if self.after_drop(x, y) >= self.max_n:
            self.end = True
            self.winner = self.turn
        self.turn = self.change_player()

    def after_drop(self, x, y):
        direct = [[[-1, -1], [1, 1]], [[-1, 0], [1, 0]], [[0, -1], [0, 1]], [[-1, 1], [1, -1]]]
        max_line = 0
        for dir in direct:
            a = self.count_block(x, y, dir[0][0], dir[0][1])
            b = self.count_block(x, y, dir[1][0], dir[1][1])
            if a + b + 1 > max_line:
                max_line = a + b + 1
        return max_line

    def count_block(self, x, y, dir_x, dir_y):
        temp_x = x + dir_x
        temp_y = y + dir_y
        count = 0
        while self.row > temp_x >= 0 and self.col > temp_y >= 0 and count < self.max_n - 1:
            if self.board[temp_x][temp_y] == self.turn:
                count += 1
                temp_x += dir_x
                temp_y += dir_y
            else:
                break
        return count


if __name__ == '__main__':
    chessboard = Board()
    chessboard.step(0)
    chessboard.step(1)
    chessboard.step(2)

    chessboard.render()
    print(chessboard.valid_moves)

    chessboard.rotate()
    chessboard.render()
    print(chessboard.valid_moves)

    chessboard.mirror()
    chessboard.render()
    print(chessboard.valid_moves)

    # while not chessboard.end:
    #     chessboard.render()
    #     x = int(input("enter drop location x(start with 0):"))
    #     y = int(input("enter drop location y(start with 0):"))
    #     action = x * chessboard.col + y
    #     chessboard.step(action)
    # chessboard.render()
    # print(chessboard.get_board())
