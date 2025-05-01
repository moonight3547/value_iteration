import gym
from gym import spaces
import numpy as np
from collections import defaultdict
import warnings

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
warnings.filterwarnings('ignore')

class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, k = 3):
        super(TicTacToeEnv, self).__init__()
        # Observation space: each cell can be -1 (O), 0 (empty), or 1 (X)
        # We represent the board as a discrete observation by flattening and converting to a string
        self.k = k
        self.box_size = k * k
        # 3^9 = 19,683 possible states when k = 3; 3^16 = 43,046,721 when k = 4.
        self.observation_space = spaces.Discrete(3**self.box_size)  

        # Action space: 9/16 possible positions (0-8/15)
        self.action_space = spaces.Discrete(self.box_size)

        # Game state tracking
        self.board = np.zeros((k, k), dtype=np.int32)
        self.done = None
        self.current_player = None
        
        # Create mapping between board states and discrete observations
        self.state_to_id = defaultdict(int)
        self._initialize_state_mapping()

    def _initialize_state_mapping(self):
        """Precompute all possible board states and assign unique IDs."""
        from itertools import product
        states = product([-1, 0, 1], repeat=self.box_size)
        for i, state in enumerate(states):
            self.state_to_id[state] = i

    def _get_observation(self):
        """Convert the board state to a discrete observation ID."""
        return self.state_to_id[tuple(self.board.flatten())]

    def reset(self,seed = None):
        super().reset(seed=seed)
        """Reset the game state and return initial observation."""
        self.board = np.zeros((self.k, self.k), dtype=np.int32)
        self.done = False
        self.current_player = 1  # Agent starts first
        return self._get_observation()
    
    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, {}
            
        row, col = action // self.k, action % self.k
        reward = 0
        info = {'valid': True}

        # Agent's turn
        if self.current_player == 1:
            if self.board[row, col] != 0:
                # Invalid move
                reward = -10
                info['valid'] = False
                return self._get_observation(), reward, False, info
            
            # Apply valid move
            self.board[row, col] = 1
            self.current_player = -1  # Switch to opponent
            # Check game state
            if self._check_win(1):
                self.done = True
                reward = 1
            elif self._is_full():
                self.done = True
                reward = 0
            else:
                # Opponent's turn
                self._opponent_move()
                self.current_player = 1  # Switch back to agent
                # Check game state after opponent move
                if self._check_win(-1):
                    self.done = True
                    reward = -1
                elif self._is_full():
                    self.done = True
                    reward = 0
        return self._get_observation(), reward, self.done, info

    def render(self, mode='human'):
        symbols = {1: 'X', -1: 'O', 0: '.'}
        print("Current board:")
        for row in self.board:
            print(' '.join([symbols[cell] for cell in row]))
        print(f"Current player: {'X (Agent)' if self.current_player == 1 else 'O (Opponent)'}")
        print()

    def _check_win(self, player):
        # Check rows
        for row in self.board:
            if all(cell == player for cell in row):
                return True
        # Check columns
        for col in self.board.T:
            if all(cell == player for cell in col):
                return True
        # Check diagonals
        if all(self.board[i, i] == player for i in range(self.k)):
            return True
        if all(self.board[i, self.k-1-i] == player for i in range(self.k)):
            return True
        return False
    
    def _is_full(self):
        """Check if the board is full."""
        return 0 not in self.board
    
    def _opponent_move(self):
        """Make a random move for the opponent."""
        empty = list(zip(*np.where(self.board == 0)))
        if empty:
            row, col = empty[np.random.choice(len(empty))]
            self.board[row, col] = -1

# Register environment
from gym.envs.registration import register
register(
    id="TicTacToe-v0",
    entry_point='__main__:TicTacToeEnv',
    kwargs={'k': 3}
)
register(
    id="TicTacToe-v1",
    entry_point='__main__:TicTacToeEnv',
    kwargs={'k': 4}
)

#Example
if __name__ == "__main__":
    env = gym.make("TicTacToe-v0")
    state = env.reset()
    env.render()
    moves = [0, 4, 1, 3, 2, 5, 6, 7, 8, 9]  # Creates three Xs in top row
    rewards = 0
    for move in moves:
        state, step_reward, done, _ = env.step(move)
        env.render()
        print(f"Reward: {step_reward}, Done: {done}\n")
        rewards += step_reward
        if done: break
    print('total rewards:',rewards)