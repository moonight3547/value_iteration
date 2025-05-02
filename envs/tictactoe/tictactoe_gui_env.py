import sys
sys.path.append("../../")

import gym
from gym import spaces
import numpy as np
from collections import defaultdict
from itertools import product
from envs.tictactoe.tictactoe_gui import *
from value_iteration import *

# X:1; O:2
class TicTacToeGUIEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self,size,player):
        super(TicTacToeGUIEnv, self).__init__()
        self.row = self.column = size
        # Action space: 9 possible positions (0-8)
        self.action_space = spaces.Discrete(self.row*self.row)
        
        # Observation space: each cell can be 2(O), 0 (empty), or 1 (X)
        # We represent the board as a discrete observation by flattening and converting to a string
        self.observation_space = spaces.Discrete(3**(self.row*self.row))  # 19,683 possible states
        self.P = defaultdict(dict)
        self.player = player
        # Game state tracking
        self.board = np.zeros((self.row, self.column), dtype=np.int32)
        self.done = None
        self.current_player = None
        self.op_move = []    
        # Create mapping between board states and discrete observations
        self.state_to_id = defaultdict(int)
        self.initialize_state_mapping()
        self.P = defaultdict(lambda: defaultdict(list))
        self.transition()

    def initialize_state_mapping(self):
        """Precompute all possible board states and assign unique IDs."""

        states = product([0, 1, 2], repeat=self.row*self.column)
        for i, state in enumerate(states):
            self.state_to_id[state] = i
#        print('id',self.state_to_id[(2,1,0,0,1,1,0,2,0)])

    def _get_observation(self):
        """Convert the board state to a discrete observation ID."""
        return self.state_to_id[tuple(self.board.flatten())]
    def transition(self):
        """Precompute all possible state transitions"""       
        # Generate all possible board states
        # count = 0
        for state in product([0, 1,2], repeat=self.row*self.column):
            # count+=1
            # print(count)
            if self.player == 1 and state.count(1) != state.count(2):
                continue
            if self.player == 2 and state.count(1) != state.count(2)+1:
                continue
            state = np.array(state).reshape(self.row, self.column)
            state_key = self._state_to_key(state)

            
            # Skip terminal states
            if self._check_win(state,1) or self._check_win(state,2) or self._is_full(state):
                continue
            # For each possible action in this state
            for action in range(self.row*self.column):
                row, col = action // self.row, action % self.column
                if state[row, col] != 0:  # Invalid move
                    self.P[state_key][action] = [
                        (1.0, state_key, -10, False)
                    ]
                    continue
                    
                # Simulate valid moves
                next_state = state.copy()
                next_state[row, col] = self.player  # Current player (agent)
                
                # Check if agent wins
                if self._check_win(next_state, self.player):
                    self.P[state_key][action] = [
                        (1.0, self._state_to_key(next_state), 1.0, True)
                    ]
                    continue
                    
                # Check if board is full
                if self._is_full(next_state):
                    self.P[state_key][action] = [
                        (1.0, self._state_to_key(next_state), 0.0, True)
                    ]
                    continue
                    
                # Simulate opponent random moves
                empty_cells = list(zip(*np.where(next_state == 0)))
                opponent_moves = []
                
                for opp_row, opp_col in empty_cells:
                    opp_state = next_state.copy()
                    opp_state[opp_row, opp_col] = 3-self.player
                    
                    # Calculate opponent move probability (uniform random)
                    prob = 1.0 / len(empty_cells)
                    
                    # Determine outcome
                    if self._check_win(opp_state, 3-self.player):
                        opponent_moves.append((prob, self._state_to_key(opp_state), -1.0, True))
                    elif self._is_full(opp_state):
                        opponent_moves.append((prob, self._state_to_key(opp_state), 0.0, True))
                    else:
                        opponent_moves.append((prob, self._state_to_key(opp_state), 0.0, False))
                
                self.P[state_key][action] = opponent_moves
    def _state_to_key(self, state):
        """Convert state array to hashable key"""
        return self.state_to_id[tuple(state.flatten())]
    def get_transitions(self, state, action):
        """Get possible transitions for a state-action pair"""
        state_key = tuple(state.flatten())
        return self.P[state_key][action]

    def reset(self,seed = None):
        super().reset(seed=seed)
        """Reset the game state and return initial observation."""
        self.board = np.zeros((self.row, self.column), dtype=np.int32)
        if self.player == 2:
            random_row = np.random.randint(0, self.row)
            random_col = np.random.randint(0, self.column)

            self.board[random_row, random_col] = 1 

        self.done = False
        self.current_player = 1  # Agent starts first
        return self._get_observation(),{}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, True,{}
            
        row, col = action // self.row, action % self.column
        reward = 0
        info = {'valid': True}

        # Agent's turn
        if self.current_player == 1:
            if self.board[row, col] != 0:
                # Invalid move
                reward = -10
                info['valid'] = False
                return self._get_observation(), reward, False, False, info
            
            # Apply valid move
            self.board[row, col] = self.player
            self.current_player = -1  # Switch to opponent
            # Check game state
            if self._check_win(self.board,self.player):
                self.done = True
                reward = 1
            elif self._is_full(self.board):
                self.done = True
                reward = 0
            else:
                # Opponent's turn
                row,col = self.opponent_move()
                self.save_opponent_move(row,col)
                self.current_player = 1  # Switch back to agent
                # Check game state after opponent move
                if self._check_win(self.board,3-self.player):
                    self.done = True
                    reward = -1
                elif self._is_full(self.board):
                    self.done = True
                    reward = 0

        return self._get_observation(), reward, self.done, False,info

    def render(self, mode='human'):
        symbols = {1: 'X', 2: 'O', 0: '.'}
        print("Current board:")
        for row in self.board:
            print(' '.join([symbols[cell] for cell in row]))
        print(f"Current player: {'X (Agent)' if self.current_player == 1 else 'O (Opponent)'}")
        print()
    def gui_render(self,policy,policy2,mode):
        game = TicTacToeGUI(policy,n=self.row,mode = mode,op_policy=policy2)
        game.initialize_state_mapping()
        game.draw()
        game.screen.onclick(game.play)
        game.screen.mainloop()
    def _check_win(self, board,player):
        # Check rows
        for row in board:
            if all(cell == player for cell in row):
                return True
        # Check columns
        for col in board.T:
            if all(cell == player for cell in col):
                return True
        # Check diagonals
        if all(board[i, i] == player for i in range(self.row)):
            return True
        if all(board[i, self.row-1-i] == player for i in range(self.column)):
            return True
        return False
    
    def _is_full(self,board):
        """Check if the board is full."""
        return 0 not in board
    
    def opponent_move(self):
        """Make a random move for the opponent."""
        empty = list(zip(*np.where(self.board == 0)))
        if empty:
            row, col = empty[np.random.choice(len(empty))]
            self.board[row, col] = 2
            return row,col
    def save_opponent_move(self,row,col):
        idx = row*self.row+col
        self.op_move.append(idx)
        return self.op_move

# Register environment
from gym.envs.registration import register
register(
    id='TicTacToeGUI-v0',
    entry_point='__main__:TicTacToeGUIEnv',
    kwargs={'size': 3}
)

#Example
if __name__ == '__main__':

    env1 = gym.make('TicTacToeGUI-v0',size = 3,player = 1)
    VI1 = ValueIterationAgent(env1, gamma=1, iters=10000, eval_iters=100, eps=1e-10, seed=233333)
    VI1.value_iteration()
    VI1.get_policy()
#    print(VI.value)
#    print(VI.policy)
    rewards = 0
    done = False
    # while not done:
    #     state, step_reward, done, trun,_ = env.step(VI.policy[env.state_to_id[tuple(env.board.flatten())]])
    #     env.render()
    #     print(f"Reward: {step_reward}, Done: {done}\n")
    #     rewards += step_reward
    #     if done: break
    # print('total rewards:',rewards)
    policy2 = env1.op_move

    env1.gui_render(VI1.policy,policy2,mode = 'human')