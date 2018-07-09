import random
import copy
import Perceptron as percept
from GameTree import Node
import time

def BuildGameTree():
    
    b=Board()
    b.board=[[0,0,0],[0,0,0],[0,0,0]]
    
    p = Node(element = [[0,0,0],[0,0,0],[0,0,0]])
    Display(p.element)
    BuildValidMoves(p)

def BuildValidMoves(p): # build the valid children of each board configuration
    turn = None # find out whose turn is it
    if sum(p.element[0])+sum(p.element[1])+sum(p.element[2]) == 0: # if x turn
        turn = 1
    else: # if o turn
        turn = -1
        
    counter = 0 # index of each board configuration in the tree
    for i in range(3):
        for j in range(3):
            if p.element[i][j] == 0 and p.element[i][j] != None and Complete(p.element) == False: # if the tile is unoccupied and the game is unresolved
                p.setChild( Node( element = copy.deepcopy(p.element) )) # copy the state of the parent board
                p.children[counter].element[i][j] += turn # add the next move
                BuildValidMoves(p.children[counter]) # recursively build onto child
                
            else:
                p.setChild( Node (element=None) ) # not a valid move
            counter += 1

def MinMax(p):
    # minmax=[0,0,0,0,0,0,0,0,0]
    for i in range(9):
        if p.Children()[i].isLeaf():
            # minmax[i] += Winner(p.Children()[i].element)
            GimmeThePoints(p.Children()[i], i, Winner(p.Children()[i].element))
            # [x+y for x,y in zip(minmax,MinMax(p.Children()[i]))]
        else:
            MinMax(p.Children()[i])
    return p

def GimmeThePoints(p, i, winner): # do we need this?
    p.util[i] += 1 if winner==1 else -1 if winner==-1 else 0
    
    if p.parent != None:
        GimmeThePoints(p.parent, i, winner)

def FlipBoard(board):
    for i in range(3):
        for j in range(3):
            if board[i][j] == 1:
                board[i][j] = -1
            elif board[i][j] == -1:
                board[i][j] = 1
    return board

def Display(p): # displays the board
    board = p.element
    
    print(" -------------   \n | {} | {} | {} | \n -------------   \n | {} | {} | {} | \n -------------   \n | {} | {} | {} | \n ------------- "
        .format(board[0][0], board[0][1], board[0][2],
                board[1][0], board[1][1], board[1][2],
                board[2][0], board[2][1], board[2][2]))


def Complete(board): # checks for completion, by victory or by tie
    completed_squares = 0
    counter = 0
    for i in range(3):
        for j in range(3):
            if board[i][j] != 0:
                completed_squares += 1
            counter += 1
            
    if completed_squares == 9: # if every position on the board is occupied
        return True
    else:
        if completed_squares > 4: # checking for a winner along the diagonal
            if Winner(board) == 0:
                return False
            else:
                return True
        else:
            return False
        
def Winner(board): # tells the color of the winner
    
    if board == None:
        return 0
    else:
        if board[0][0] == board[1][1] == board[2][2] != 0 or board[0][2] == board[1][1] == board[2][0] != 0:
            return board[1][1]
        
        c=0
        while c < 3: # checking for a winner along the horizontal and vertical
            if board[c][0] == board[c][1] == board[c][2] != 0:
                return board[c][1]
            if board[0][c] == board[1][c] == board[2][c] != 0:
                return board[1][c]
            c += 1
        return 0
    
    
            
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

class HumanPlayer():
    def __init__(self, side=None):
        self.side=side

    def move(self,g):

        pos=input('Enter a position: ')

        if (pos.lower() != 'q' and pos.lower() != 'quit'):
            try:
                s = g.add(self.side,int(pos))

                if (s == False):
                    print("This move cannot be made")
                    self.move(g)
            
            except ValueError:
                print("Not a number. Try again")
                self.move(g)
        else:
            g.winner=0

# x's job is to pick the highest scoring move (max)
# y's job is to pick the lowest scoring move (min)

class PerfectPlayer():
    def __init__(self, side=None, curr_pos=None):
        self.side=side
        # self.tree=tree
        self.curr_pos = curr_pos

    def move(self,g):
        
        print(self.curr_pos)
        
        prev_b = self.curr_pos.element
        prev_b = prev_b[0] + prev_b[1] + prev_b[2]
        new_b = []
        
        for i in range(len(g.board)):
            for j in range(len(g.board[i])):
                new_b.append(g.board[i][j])
                
        print(prev_b)
        print(new_b)
        
        p=[abs(prev_b-new_b) for prev_b,new_b in zip(prev_b,new_b)]
        print(p)
        
        self.curr_pos = self.curr_pos.Children()[p.index(max(p))]
        
        print(self.curr_pos)
        print(self.curr_pos.util)
        
        util = self.curr_pos.util[:] # utility we are considering
        
        while sum(util) != 0:
            
            if prev_b[util.index(max(util))] == 0:
                next_move = util.index(max(util))
            
                if self.side == -1: # playing o
                    g.add(-1,next_move+1)
                    print("Move: ", next_move)
                    print("Pos: ", self.curr_pos.Children())
                    self.curr_pos = self.curr_pos.Children()[next_move]
                    
                elif self.side == 1:
                    g.add(1,next_move+1)
                    self.curr_pos = self.curr_pos.Children()[next_move]
                    
                # self.curr_pos = self.curr_pos.Children()[p.index(max(p))]
                    
            else:
                util[util.index(max(util))] = 0
            

class NeuralNetPlayer():
    def __init__(self, side=None, n=None):
        self.side=side
        self.n=n
        
    def move(self,g):
        v=[]
        
        

        for i in range(len(g.board)): # convert the matrix into a vector
            for j in range(len(g.board[i])):
                v.append(g.board[i][j])

        result = self.n.feedForward(v)
        result = list(result)
        result = [ float('%.3f' % elem) for elem in result ]
        print(result)

        while True:
            s = g.add(self.side, result.index(max(result))+1) # try the max value--if it is taken, take the next best one
            if s == False:
                result[result.index(max(result))] = 0.0
            else:
                break
        
class RandomPlayer():
    def __init__(self, side=None):
        self.side=side

    def move(self,g):
        s = g.add(self.side,random.randint(1,9))
        if (s == False):
            self.move(g)
        
def PlayGame(p1,p2,g):
    while True:
        if (g.winner == None):
            p1.move(g)
            print("Count: ", g.count)
            g.display()
        if (g.winner == None):
            p2.move(g)
            print("Count: ", g.count)
            g.display()
        else:
            break

# def Test():
    # n = percept.NN([9,10,10,10,10,9])
    
p = Node(element = [[0,0,0],[0,0,0],[0,0,0]])
print("Building game tree...")
t_1=time.time()
BuildValidMoves(p)
t_2=time.time()
print("Time (s): ", t_2-t_1)
p=MinMax(p)

h=HumanPlayer(1)
n=PerfectPlayer(-1,p)
print(type(p))
g=Board()
g.reset()
g.display()
PlayGame(h,n,g) # player is currently going first


'''
then = time.time()
p = Node(element = [[0,0,0],[0,0,0],[0,0,0]])
# p = Node(element = [[1,-1,1],[1,-1,0],[-1,0,0]])
# g = Node(element = [[-1,1,-1],[-1,1,0],[1,0,0]])
print("Building game tree...")
BuildValidMoves(p)
# BuildValidMoves(g)
print("Calculating minmax assignments...")
m=MinMax(p)
Display(p)
print(m)
now = time.time()
print("Time elapsed: ", now-then)
'''
# 
# print(MinMax(g))