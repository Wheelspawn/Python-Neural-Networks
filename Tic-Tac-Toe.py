import random
from GameTree import *

def BuildGameTree():
    l = []
    b=Board()
    b.board=[[0,0,0],[0,0,0],[0,0,0]]
    
    for i in range(0,3):
        for j in range(0,3):
            new = [[0,0,0],[0,0,0],[0,0,0]]
            new[i][j] += 1
            l.append(new)
    return l

class Board():

    conv = {-1:'o', 0:' ', 1:'x'} # dictionary that turns the integers into x and o
    
    def __init__(self, count=0, board=[[0,0,0],[0,0,0],[0,0,0]],winner=None):
        self.count=count
        self.board=board
        self.winner=winner
        
    def display(self): # displays the board
        print(" -------------   \n | {} | {} | {} | \n -------------   \n | {} | {} | {} | \n -------------   \n | {} | {} | {} | \n ------------- "
        .format(self.conv[self.board[0][0]], self.conv[self.board[0][1]], self.conv[self.board[0][2]],
                self.conv[self.board[1][0]], self.conv[self.board[1][1]], self.conv[self.board[1][2]],
                self.conv[self.board[2][0]], self.conv[self.board[2][1]], self.conv[self.board[2][2]]))

    def add(self, char, n):
        n-=1
        a = n//3 # integer ops for getting the right position on the board
        b = n%3
        if n < 9:
            if self.board[a][b] != 1 and self.board[a][b] != -1: # if the space on the board has not yet been taken
                self.board[a][b] = char # add the move
                self.count += 1
            else:
                return False # flag: invalid move
        else:
            return False # flag: board is already filled
            
        if self.count > 4: # checking for a winner along the diagonal
            if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0 or self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
                print(char, " is winner")
                self.winner=char
            
            c=0
            while c < 3: # checking for a winner along the horizontal and vertical
                if self.board[c][0] == self.board[c][1] == self.board[c][2] != 0 or self.board[0][c] == self.board[1][c] == self.board[2][c] != 0:
                    print(char, " is winner")
                    self.winner=char
                    break
                
                else:
                    c+=1
                    
        if self.count == 9:
            print('tie')
            self.winner=0

    def reset(self):
        self.count=0
        self.board=[[0,0,0],[0,0,0],[0,0,0]]
        self.winner=None
