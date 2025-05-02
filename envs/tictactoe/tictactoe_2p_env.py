import sys
sys.path.append("../../")

import gym
from gym import spaces
import numpy as np
from collections import defaultdict
from itertools import product
from value_iteration import *

import warnings

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
warnings.filterwarnings('ignore')

# X:1; O:2
class TicTacToeEnv2(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self,size):
        super(TicTacToeEnv2, self).__init__()
        self.row, self.column = size, size
        # Action space: 9 possible positions (0-8)
        self.num_actions = self.row*self.column
        self.action_space = spaces.Discrete(self.num_actions)
        self.num_states  = 3**self.num_actions
        self.observation_space = spaces.Discrete(self.num_states)  # 19,683 possible states for 3*3
        self.P = defaultdict(dict)
        # Game state tracking
        # self.board = np.zeros((self.row, self.column), dtype=np.int32)
        # Create mapping between board states and discrete observations
        self.state_to_id = defaultdict(int)
        self.id_to_state = [None for _ in range(self.num_states)]
        self.state_type = np.zeros(3 ** (self.row*self.column), dtype = int)
        self.initialize_state_mapping()
#        self.reward = np.zeros((self.observation_space.n, self.action_space.n))
        self.reward = {}
        self.get_rewards()

    def initialize_state_mapping(self):
        """Precompute all possible board states and assign unique IDs."""

        states = product([0, 1, 2], repeat=self.row*self.column)
        for i, state in enumerate(states):
            board = tuple(state)
            delta = board.count(1) - board.count(2)
            count0 = board.count(0)
            board = np.array(board).reshape(self.row, self.column)
            win_count1 = self._check_win(board, 1)
            win_count2 = self._check_win(board, 2)
            if (win_count1 + win_count2 > 2) or (win_count1 and win_count2):
                self.state_type[i] = 0
                continue
            if delta == 0:
                self.state_type[i] = 3 if (win_count2 or not count0) else (0 if win_count1 else 1)
            elif delta == 1:
                self.state_type[i] = 3 if (win_count1 or not count0) else (0 if win_count2 else 2)
            else :
                self.state_type[i] = 0
                continue
            self.state_to_id[state] = i
            self.id_to_state[i] = board

    def _get_observation(self):
        """Convert the board state to a discrete observation ID."""
        return self.state_to_id[tuple(self.board.flatten())]
        
    # state1 = env.get_state(2, state2, action2)
    def get_state(self,player,state,action):
        assert self.state_type[state] == player
        board = self.id_to_state[state].copy()
        row, col = action // self.row, action % self.column
        if board[row,col] == 0: 
            board[row,col] = player
            assert self.id_to_state[state][row, col] != player
        return self.state_to_id[tuple(board.flatten())]
    
    def get_rewards(self):
        for state in range(self.observation_space.n):
            player = self.state_type[state]
            if player not in {1, 2}: continue
            self.reward[state] = np.zeros(self.num_actions)
            board = self.id_to_state[state]
            for action in range(self.action_space.n):
                row, col = action // self.row, action % self.row
                if board[row,col] :
                    self.reward[state][action] = -100
                else :
                    board[row,col] = player
                    if self._check_win(board, player):
                        self.reward[state][action] = 1
                    else:
                        r = 0
                        for action2 in range(self.action_space.n):
                            x, y = action2 // self.row, action2 % self.row
                            if board[x, y]: continue
                            board[x, y] = 3 - player
                            if self._check_win(board, 3 - player):
                                r = -1
                                board[x, y] = 0
                                break
                            board[x, y] = 0
                        self.reward[state][action] = r
                    board[row,col] = 0

    def _check_win(self, board,player):
        # Check rows
        sum = 0
        for row in board:
            if all(cell == player for cell in row):
                sum += 1
        # Check columns
        for col in board.T:
            if all(cell == player for cell in col):
                sum += 1
        # Check diagonals
        if all(board[i, i] == player for i in range(self.row)):
            sum += 1
        if all(board[i, self.row-1-i] == player for i in range(self.column)):
            sum += 1
        return sum
    
    def _is_full(self,board):
        """Check if the board is full."""
        return 0 not in board

def two_agents(env, agent1, agent2, gamma = 1, iters = 100, eps = 1e-5):
    # env follow gym.Env setting
    # agent1, agent2 are two value iteration agent
    
    # value iteration of two agents
    # Start Iteration
    for state in range(env.num_states):
        if env.state_type[state] != 1: agent1.value[state] = 0
        if env.state_type[state] != 2: agent2.value[state] = 0
    
    for i in range(iters):
        prev_value1 = agent1.value.copy()
        prev_value2 = agent2.value.copy()
        for state1 in range(agent1.num_states):
            if env.state_type[state1] != 1: continue
            Q = []
            for action1 in range(agent1.num_actions):
                state2 = env.get_state(1, state1, action1)
                if state2 == state1:
                    Q.append(-100 + gamma * prev_value1[state1])
                elif env.state_type[state2] == 3:
                    Q.append(env.reward[state1][action1])
                else :
                    state_next = env.get_state(2, state2, agent2.policy[state2])
                    Q.append(env.reward[state1][action1] + gamma * prev_value1[state_next])
            agent1.Q[state1] = Q
            agent1.value[state1] = np.max(Q)
            agent1.policy[state1] = np.argmax(Q)

        for state2 in range(agent2.num_states):
            if env.state_type[state2] != 2: continue
            Q = []
            for action2 in range(agent2.num_actions):
                state1 = env.get_state(2, state2, action2) # implement this function with params: player (1/2), state, action
                if state1 == state2:
                    Q.append(-100 + gamma * prev_value2[state2])
                elif env.state_type[state1] == 3:
                    Q.append(env.reward[state2][action2])
                else :
                    state_next = env.get_state(1, state1, agent1.policy[state1])
                    Q.append(env.reward[state2][action2] + gamma * prev_value2[state_next])
#                state_next = state2 if state1 == state2 else env.get_state(1, state1, agent1.policy[state1])
#                Q.append(env.reward[state2][action2] + gamma * prev_value2[state_next])
            agent2.Q[state2] = Q
            agent2.value[state2] = np.max(Q)
            agent2.policy[state2] = np.argmax(Q)
        delta1 = np.sum(np.fabs(prev_value1 - agent1.value))
        delta2 = np.sum(np.fabs(prev_value2 - agent2.value))
        print(f"iteration {i+1}: player1 value delta {delta1}, player2 value delta: {delta2}")
        if (delta1 <= eps and delta2 <= eps):
            print("Converged in %d Iterations"%(i))
            return agent1.value, agent2.value, agent1.policy, agent2.policy
    print("\033[31m[Warning]\033[0m Iterated over %d Iterations and couldn't converge"%(iters))
    return agent1.value, agent2.value, agent1.policy, agent2.policy

def run_NE(env_new, VI1, VI2):
    board = np.array([[0 for j in range(3)] for i in range(3)])
    game_board = [['.' for j in range(3)] for i in range(3)]
    state = env_new.state_to_id[tuple(board.flatten())]
    player = 1
    for t in range(9):
        if player == 1:
            action = VI1.policy[state]
            state1 = env_new.get_state(1, state, action)
            x, y = action // 3, action % 3
            r = env_new.reward[state][action]
            board[x][y] = 1
            game_board[x][y] = 'X'
            player = 2
        else:
            action = VI2.policy[state]
            state1 = env_new.get_state(2, state, action)
            x, y = action // 3, action % 3
            r = env_new.reward[state][action]
            board[x][y] = 2
            game_board[x][y] = 'O'
            player = 1
        print(f"Step {t+1}, state {state} Player {3 - player} moves ({x}, {y}) reward {r, env_new.reward[state][x*3+y]}: ")
        state = state1
        for i in range(3):
            print(' '.join(game_board[i]))