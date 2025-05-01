import gym
from gym import spaces
import numpy as np
import warnings
warnings.filterwarnings('ignore')


import gym
import numpy as np
from gym import spaces

class ZeroSumEnv(gym.Env):
    def __init__(self, payoff_matrix, player, strategy='pure',max_steps=1):
        super(ZeroSumEnv, self).__init__()
        
        # payoff matrixA ∈ R^{m×n}
        self.A = payoff_matrix
        self.m, self.n = payoff_matrix.shape
        self.P = {}        
        self.player = player
        self.strategy = strategy
        if self.strategy == 'pure':
            if player == 1:
                self.action_space = spaces.Discrete(self.m)
            else:
                self.action_space = spaces.Discrete(self.n)
        elif self.strategy == 'mixed':
            self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Discrete(max_steps)
        self.max_steps = max_steps
        self.current_step = 0
        if self.strategy == 'pure':
            if self.player == 1:
                self.transition_1()
            elif self.player == 2:
                self.transition_2()
        else:
            self.transition_mixed()
    def get_action(self,action):
        if self.strategy == 'pure':

            if self.player ==1:
                opponent_action = np.argmax(A[action])
                return opponent_action
            else:
                opponent_action = np.argmin(A[:,action])
                return opponent_action
        else:
            opponent_action = 10-action
            return opponent_action
    def transition_1(self):
        for state in range(self.observation_space.n):
            self.P[state] = {}
            for action in range(self.action_space.n):
                j  = self.get_action(action)

                reward = -self.A[action][j]
                self.P[state][action] = [(1.0, state, reward, False)]
    def transition_2(self):
        for state in range(self.observation_space.n):
            self.P[state] = {}
            for action in range(self.action_space.n):
                j  = self.get_action(action)
                reward = self.A[j][action]
                self.P[state][action] = [(1.0, state, reward, False)]        
    def transition_mixed(self):
        for state in range(self.observation_space.n):
            self.P[state] = {}
            for action in range(self.action_space.n):
                prob = action/10
                reward = np.min(np.array([prob,1-prob]).reshape(1,2)@self.A)
                self.P[state][action] = [(1.0, state, reward, False)]  
    def reset(self,seed = None):
        self.current_step = 0
        info = {'Info:':False}
        return self.current_step,info
    
    def step(self, actions):
        if self.strategy == 'pure':
            if self.player == 1:
                j = self.get_action(actions)  # Player 1:i，Player 2:j    
                # calculate the reward
                reward_p1 = self.A[actions][j]     
                self.current_step += 1
                done = (self.current_step >= self.max_steps)
                

                return self.current_step, reward_p1, done, False,{}
            else:
                i = self.get_action(actions)  # Player 1:i，Player 2:j    
                # calculate the reward
                reward_p2 = self.A[i][actions]    
                self.current_step += 1
                done = (self.current_step >= self.max_steps)
                return self.current_step, reward_p2, done, False,{}   
        else:
            if self.player == 1:
                prob = actions/10
                reward = np.max(np.array([prob,1-prob]).reshape(1,2)@self.A)        
                self.current_step += 1
                done = (self.current_step >= self.max_steps)
                return self.current_step, reward, done, False,{}     
            else:
                prob = actions/10
                reward = np.max(self.A@np.array([prob,1-prob]).reshape(2,1))        
                self.current_step += 1
                done = (self.current_step >= self.max_steps)
                return self.current_step, reward, done, False,{}                 
    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Payoff Matrix: \n{self.A}")


from gym.envs.registration import register
register(
    id='Zero-Sum',
    entry_point='__main__:ZeroSumEnv',
)
from value_iteration import *
if __name__ == "__main__":
    A= np.array([
[1,-1],
[-1,1]
    ])
    env1 = gym.make('Zero-Sum',payoff_matrix = A, player = 1,strategy = 'mixed')
    env2 = gym.make('Zero-Sum',payoff_matrix = A, player = 2,strategy = 'mixed')
    VI1 = ValueIterationAgent(env1, gamma=1, iters=10000, eval_iters=100, eps=1e-5, seed=233333)
    VI1.run()
    VI2 = ValueIterationAgent(env2, gamma=1, iters=10000, eval_iters=100, eps=1e-5, seed=233333)
    VI2.run()
    print("Optimal Policy: ", [VI1.policy[0]/10,1-VI1.policy[0]/10])
    print("Optimal Value: ", VI1.value[0])
    print("Optimal Policy: ", [VI2.policy[0]/10,1-VI1.policy[0]/10])
    print("Optimal Value: ", VI2.value[0])
