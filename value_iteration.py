import time
import numpy as np
from matplotlib import pyplot as plt

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

class ValueIterationAgent:
    def __init__(self, env, gamma=1.0, iters=10000, eval_iters=100, eps=1e-20, seed=3547):
        self.env = env
        self.env.reset(seed=seed)
        self.gamma = gamma
        self.iters = iters
        self.eval_iters = eval_iters
        self.eps = eps
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.time_cost = "Unknown"

    def value_iteration(self):
        self.value = np.zeros(self.num_states)
        # Start Iteration
        for i in range(self.iters):
            prev_value = self.value.copy()
            for state in range(self.num_states):
                Q = []
                for action in range(self.num_actions):
                    outcomes = []
                    for prob, next_state, reward, done in self.env.P[state][action]:
                        outcomes.append(prob * (reward + self.gamma * prev_value[next_state]))
                    Q.append(np.sum(outcomes))
                self.Q[state] = Q
                self.value[state] = np.max(Q)
            if (np.sum(np.fabs(prev_value - self.value)) <= self.eps):
                print("Converged in %d Iterations"%(i))
                return self.value
        print("\033[31m[Warning]\033[0m Iterated over %d Iterations and couldn't converge"%(self.iters))
    
    def get_policy(self):
        self.policy = np.zeros(self.num_states, dtype=int)
        for state in range(self.num_states):
            Q = []
            for action in range(self.num_actions):
                outcomes = []
                for prob, next_state, reward, done in self.env.P[state][action]:
                    outcomes.append(prob * (reward + self.gamma * self.value[next_state]))
                Q.append(np.sum(outcomes))
            self.Q[state] = Q
            self.policy[state] = np.argmax(Q)

    def get_trajectory(self):
        tot_reward = 0
        discount = 1
        step, done = 0, False
        state, _ = self.env.reset()
        while not done and discount > self.eps:
            action = self.policy[state]
            state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            tot_reward += reward * discount
            discount *= self.gamma
            step += 1
        return tot_reward

    def eval_policy(self):
        scores = [self.get_trajectory() for _ in range(self.eval_iters)]
        self.mean_score = np.mean(scores)
        self.std_score = np.std(scores)
        self.best_score = np.max(scores)
        print("Mean score = %0.2f. Standard Deviation = %0.2f. Best score = %0.2f. Time taken = %4.4f seconds"%\
              (self.mean_score, self.std_score, self.best_score, self.time_cost))
    
    def run(self):
        startTime = time.time()
        self.value_iteration()
        self.get_policy()
        endTime = time.time()
        self.time_cost = endTime - startTime
        self.eval_policy()
    
    def plot_values(self, shape):
        # Reshape the value function to a 4x4 grid for visualization
        value_sq = np.reshape(self.value, shape)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        im = ax.imshow(value_sq, cmap='cool')
        for (j,i),label in np.ndenumerate(value_sq):
            ax.text(i, j, np.round(label, 5), ha='center', va='center', fontsize=14)
        plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.title('State-Value Function')
        plt.show()

class RandomValueIterationAgent(ValueIterationAgent):
    def __init__(self, env, gamma=1, iters=10000, eval_iters=100, eps=1e-20, seed=3547, ratio = 0.3, reduction = False, new_ratio = 0.02):
        super().__init__(env, gamma, iters, eval_iters, eps, seed)
        np.random.seed(seed)
        self.ratio = ratio
        self.reduction = reduction
        self.neighbor = None
        self.new_ratio = new_ratio
        if self.reduction:
            self.neighbor = [ [] for state in range(self.num_states)]
            for state in range(self.num_states):
                for action in range(self.num_actions):
                    for prob, next_state, reward, done in self.env.P[state][action]:
                        if (prob * self.num_states > eps) :
                            self.neighbor[next_state].append(state)

    def value_iteration(self):
        self.value = np.zeros(self.num_states)
        if self.reduction:
            available = np.zeros(self.num_states, dtype=int)
            cnt = self.num_states
            for i in range(self.iters):
                prev_value = self.value.copy()
                reduced_ratio = self.ratio * self.num_states / cnt
                cnt = 0
                for state in range(self.num_states):
                    ratio = reduced_ratio if available[state] == i else self.new_ratio
                    # Randomly skip the state with probability 1 - adjusted ratio
                    coinflip = np.random.rand()
                    if coinflip > ratio: continue
                    Q = []
                    for action in range(self.num_actions):
                        outcomes = []
                        for prob, next_state, reward, done in self.env.P[state][action]:
                            outcomes.append(prob * (reward + self.gamma * prev_value[next_state]))
                        Q.append(np.sum(outcomes))
                    self.Q[state] = Q
                    self.value[state] = np.max(Q)
                    for prev_state in self.neighbor[state]:
                        if (available[prev_state] != i+1):
                            available[prev_state] = i+1
                            cnt += 1
                if i >= 100 and (np.sum(np.fabs(prev_value - self.value)) <= self.eps * self.ratio):
                    print("Converged in %d Iterations"%(i))
                    return self.value
        else:
            for i in range(self.iters):
                prev_value = self.value.copy()
                for state in range(self.num_states):
                    # Randomly skip the state with probability (1 - ratio)
                    coinflip = np.random.rand()
                    if coinflip > self.ratio: continue
                    Q = []
                    for action in range(self.num_actions):
                        outcomes = []
                        for prob, next_state, reward, done in self.env.P[state][action]:
                            outcomes.append(prob * (reward + self.gamma * prev_value[next_state]))
                        Q.append(np.sum(outcomes))
                    self.Q[state] = Q
                    self.value[state] = np.max(Q)
                if i >= 100 and (np.sum(np.fabs(prev_value - self.value)) <= self.eps * self.ratio):
                    print("Converged in %d Iterations"%(i))
                    return self.value
        print("\033[31m[Warning]\033[0m Iterated over %d Iterations and couldn't converge"%(self.iters))        
    

class CyclicValueIterationAgent(ValueIterationAgent):
    def __init__(self, env, gamma=1.0, iters=10000, eval_iters=100, eps=1e-20, seed=3547, rand=False):
        super().__init__(env, gamma, iters, eval_iters, eps, seed)
        self.rand = rand
        np.random.seed(seed)
        self.perm = np.random.permutation(self.num_states)
    
    def value_iteration(self):
        self.value = np.zeros(self.num_states)
        # Start Iteration
        for i in range(self.iters):
            prev_value = self.value.copy()
            for s in range(self.num_states):
                state = self.perm[s] if self.rand else s
                Q = []
                for action in range(self.num_actions):
                    outcomes = []
                    for prob, next_state, reward, done in self.env.P[state][action]:
                        outcomes.append(prob * (reward + self.gamma * self.value[next_state]))
                    Q.append(np.sum(outcomes))
                self.Q[state] = Q
                self.value[state] = np.max(Q)
            if self.rand: 
                self.perm = np.random.permutation(self.num_states)
            if (np.sum(np.fabs(prev_value - self.value)) <= self.eps):
                print("Converged in %d Iterations"%(i))
                return self.value
        print("\033[31m[Warning]\033[0m Iterated over %d Iterations and couldn't converge"%(self.iters))
