
class Board():
    def __init__(self, count=0, board=[[0,0,0],[0,0,0],[0,0,0]]):
        self.count=count
        self.board=board
        
    def display(self):
        print(" -------------   \n
               | {} | {} | {} | \n
                -------------   \n
               | {} | {} | {} | \n
                -------------   \n
               | {} | {} | {} | \n
                -------------"
        .format(self.board[0][0],self.board[0][1],self.board[0][2],
                self.board[1][0],self.board[1][1],self.board[1][2],
                self.board[2][0],self.board[2][1],self.board[2][2]))

    def add(self, char, n):
        self.count += 1
        n-=1
        a = n//3
        b = n%3
        if n < 9:
            if self.board[a][b] != 'x' and self.board[a][b] != 'o':
                self.board[a][b] = char
            else:
                return False
        else:
            return False
            
        if self.count > 4:
            if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ' or self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
                print(char, " win")
            
            c=0
            while c < 3:
                if self.board[c][0] == self.board[c][1] == self.board[c][2] != ' ' or self.board[0][c] == self.board[1][c] == self.board[2][c] != ' ':
                    print(char, " win")
                    break
                else:
                    c+=1
        if self.count == 9:
            print('tie')
