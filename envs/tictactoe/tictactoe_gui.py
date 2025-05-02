from itertools import product
import json
import turtle
import numpy as np
from collections import defaultdict

class TicTacToeGUI():
    def __init__(self, policy,n,mode,op_policy=None):
        self.size = n
        self.current_player = "X"
        self.board = (0,) * n*n
        self.is_vs_computer = False  # Flag to check if playing against the computer
        self.computer_symbol = "O"   # Symbol for the computer player
        self.policy = policy
        self.op_policy = op_policy
        self.mode = mode
        # Calculate the window position to center it on the screen
        screen = turtle.Screen()
        screen.setup(800,800)
        screen.setworldcoordinates(-500,-500,500,500)
        screen.title("Tic Tac Toe")
        turtle.speed(0)
        turtle.hideturtle()
        screen.tracer(0,0)
        score = turtle.Turtle()
        score.up()
        score.hideturtle()
        self.screen = screen
        self.ROWS = self.COLS = n
        self.STARTX = self.STARTY = -450
        self.WIDTH = -2*self.STARTX
        self.HEIGHT = -2*self.STARTY
        self.turn=1
        self.working=False
        self.state_to_id = defaultdict(int)
        self.initialize_state_mapping()
    def draw_rectangle(self, x,y,w,h):
        turtle.up()
        turtle.goto(x,y)
        turtle.seth(0)
        turtle.down()
        turtle.fd(w)
        turtle.left(90)
        turtle.fd(h)
        turtle.left(90)
        turtle.fd(w)
        turtle.left(90)
        turtle.fd(h)
        turtle.left(90)
        row_gap = self.HEIGHT/self.ROWS
        col_gap = self.WIDTH/self.COLS

        for col in range(self.COLS - 1):
            turtle.up()
            turtle.goto(x + (col + 1) * col_gap, y)
            turtle.seth(90)
            turtle.down()
            turtle.fd(h)
        for row in range(self.ROWS - 1):
            turtle.up()
            turtle.goto(x, y + (row + 1) * row_gap)
            turtle.seth(0)
            turtle.down()
            turtle.fd(w)        
  
    def draw_circle(self,x,y,r,color):
        turtle.up()
        turtle.goto(x,y-r)
        turtle.seth(0)
        turtle.down()
        turtle.fillcolor(color)
        turtle.begin_fill()
        turtle.circle(r,360,150)
        turtle.end_fill()
    def draw_X( self, x,y):
        turtle.up()
        turtle.goto(x - 40,y - 40)
        turtle.seth(0)
        turtle.left(45)
        turtle.down()
        turtle.forward(150)
        turtle.back(75)
        turtle.left(90)
        turtle.forward(75)
        turtle.backward(150)
        # turtle.done()
    def init_board(self):
        board = []
        for _ in range(self.ROWS):
            # row = []
            for _ in range(self.COLS):
                board.append(0)
            # board.append(row)
        self.board = board
    def initialize_state_mapping(self):
        """Precompute all possible board states and assign unique IDs."""

        states = product([0, 1, 2], repeat=self.size*self.size)
        for i, state in enumerate(states):
            self.state_to_id[state] = i
    def draw_pieces(self):
 
        row_gap = self.HEIGHT/self.ROWS
        col_gap = self.WIDTH/self.COLS
        
        for i in range(self.ROWS):
            Y = self.STARTY + row_gap / 2 + (self.ROWS - 1 - i) * row_gap
            for j in range(self.COLS):
                X = self.STARTX + col_gap/2 + j * col_gap
                if self.board[i * self.ROWS + j] == 0:
                    pass
                elif self.board[i * self.ROWS + j] == 2:
                    self.draw_circle(X,Y,row_gap/3,'black')
                else:
                    self.draw_X(X,Y)

    def draw_board(self):
        self.draw_rectangle(self.STARTX,self.STARTY,self.WIDTH,self.HEIGHT)
    def draw(self):
        self.init_board()
        self.draw_board()
        # self.draw_pieces()
        self.screen.update()
    # For human
    def coordiate_to_index(self, x, y):
        row_gap = self.HEIGHT/self.ROWS
        col_gap = self.WIDTH/self.COLS
        col = 0
        col_threshold = self.STARTX + col_gap
        while col_threshold < x:
            col += 1
            col_threshold += col_gap
        row = self.ROWS - 1
        row_threshold = self.STARTY + row_gap
        while row_threshold < y:
            row -= 1
            row_threshold += row_gap
        
        return (row, col)
    
    def play(self,x,y):
        # if self.working: return
        # self.working = True
      
        # (row, col) = self.coordiate_to_index(x,y)
        # self.board[row * self.ROWS + col] = 2
        if self.mode == "random":
            self.make_agent_move(self.policy,1)
            self.draw_pieces()
            self.screen.update()
            if self.check_win(1) or self.check_tie():
                self.end_game()
            self.make_agent2_move(self.op_policy,2)
            self.draw_pieces()
            self.screen.update()
            if self.check_win(2) or self.check_tie():
                self.end_game()
        elif self.mode == "human":
            self.make_agent_move(self.policy,1)
            self.draw_pieces()
            self.screen.update()
            if self.check_win(1) or self.check_tie():
                self.end_game()            
            (row, col) = self.coordiate_to_index(x,y)
            self.board[row * self.ROWS + col] = 2
            self.draw_pieces()
            self.screen.update()
            
            if self.check_win(2) or self.check_tie():
                self.end_game()
        elif self.mode == 'agent':
            self.make_agent_move(self.policy,1)
            self.draw_pieces()
            self.screen.update()
            if self.check_win(1) or self.check_tie():
                self.end_game()
            self.make_agent2_move(self.op_policy)
            self.draw_pieces()
            self.screen.update()

            if self.check_win(2) or self.check_tie():
                self.end_game()



    def make_move(self, idx):
        if idx == None:
            pass
        elif self.board[idx] == 0:
            self.board[idx] = 1
    def make_random_move(self):
        state = tuple(self.board)
        empty = state.count(0)
        if empty != 0:
            zero_indices = [i for i, x in enumerate(state) if x == 0]
            idx = np.random.choice(zero_indices)
            self.board[idx] = 2
        return idx
    # Agent who play first ("X")
    def make_agent_move(self,policy,i):
        state_id = self.state_to_id[tuple(self.board)]

        self.board[(policy[state_id])]=i
    
    # Agent who play second ("O")
    def make_agent2_move(self,policy,player):
        state = tuple(self.board)
        empty = state.count(0)
        turn = int((self.size*self.size-empty-player+1)/2)
        self.board[policy[turn]] = player

    def check_win(self,player):
        board_2D = np.reshape(self.board,(self.size,self.size))
        for row in board_2D:
            if all(cell == player for cell in row):
                turtle.TK.messagebox.showinfo("Game Over", f"Player {player} wins!")
                return True
        # Check columns
        for col in board_2D.T:
            if all(cell == player for cell in col):
                turtle.TK.messagebox.showinfo("Game Over", f"Player {player} wins!")
                return True
        # Check diagonals
        if all(board_2D[i, i] == player for i in range(self.size)):
                turtle.TK.messagebox.showinfo("Game Over", f"Player {player} wins!")
                return True
        if all(board_2D[i, self.size-1-i] == player for i in range(self.size)):
                turtle.TK.messagebox.showinfo("Game Over", f"Player {player} wins!")
                return True
        return False


    def check_tie(self):
        if 0 not in self.board:
            turtle.TK.messagebox.showinfo("Game Over", "It's a tie!")
            return True
        return False

    def end_game(self):

        self.screen.clear()
        # self.restart_game()

    def restart_game(self):
        # self.screen.bye()
        self.__init__()
        self.draw()
        self.start()
        #self.start_game("X")
class RandomAgent():
    def __init__(self,size):
        self.size = size
        self.board = [0,]*size*size
    def get_policy(self):
        state = tuple(self.board)
        policy = []
        empty = state.count(0)
        while empty != 0:
            zero_indices = [i for i, x in enumerate(state) if x == 0]
            idx = np.random.choice(zero_indices)
            self.board[idx] = 2
            policy.append(idx)
            state = tuple(self.board)
            empty = state.count(0)
        return policy
        
# X:1; O:2
if __name__ == "__main__":
    # np.random.seed(42)
    policy = [0,1,3,4,8,5,6,7,8,9,2,5,4,8,2,10,15,1,4,13,12]
    policy2 = RandomAgent(3).get_policy()
    game = TicTacToeGUI(policy,n=4,mode = 'human',op_policy=policy2)
    game.draw()
    game.screen.onclick(game.play)
    game.screen.mainloop()