import random
import Perceptron as percept

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

class PerfectPlayer():
    def __init__(self, side=None):
        self.side=side

    def move(self,g):
        print("n/a")

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

def Test():
    n = percept.NN([9,10,10,10,10,9])
    h=HumanPlayer(1)
    n=NeuralNetPlayer(-1,n)
    g=Board()
    PlayGame(n,h,g)
