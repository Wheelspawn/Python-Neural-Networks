class Board():

    conv = {-1:'o', 0:' ', 1:'x'} # dictionary that turns the integers into x and o
    
    def __init__(self, count=0, board=[[0,0,0],[0,0,0],[0,0,0]]):
        self.count=count
        self.board=board
        
    def display(self): # displays the board
        print(" -------------   \n | {} | {} | {} | \n -------------   \n | {} | {} | {} | \n -------------   \n | {} | {} | {} | \n ------------- "
        .format(self.conv[self.board[0][0]], self.conv[self.board[0][1]], self.conv[self.board[0][2]],
                self.conv[self.board[1][0]], self.conv[self.board[1][1]], self.conv[self.board[1][2]],
                self.conv[self.board[2][0]], self.conv[self.board[2][1]], self.conv[self.board[2][2]]))

    def add(self, char, n):
        self.count += 1
        n-=1
        a = n//3 # integer ops for getting the right position on the board
        b = n%3
        if n < 9:
            if self.board[a][b] != 1 and self.board[a][b] != -1: # if the space on the board has not yet been taken
                self.board[a][b] = char # add the move
            else:
                return False # flag: invalid move
        else:
            return False # flag: board is already filled
            
        if self.count > 4: # checking for a winner along the diagonal
            if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ' or self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
                print(char, " is winner")
                return char
            
            c=0
            while c < 3: # checking for a winner along the horizontal and vertical
                if self.board[c][0] == self.board[c][1] == self.board[c][2] != ' ' or self.board[0][c] == self.board[1][c] == self.board[2][c] != ' ':
                    print(char, " is winner")
                    return char
                else:
                    c+=1
        if self.count == 9:
            print('tie')
