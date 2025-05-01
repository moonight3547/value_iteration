import random
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

class PolicyIterationAgent(ValueIterationAgent):
    def __init__(self, env, gamma=1, iters=10000, eval_iters=100, eps=1e-20, seed=3547):
        super().__init__(env, gamma, iters, eval_iters, eps, seed)
    
    def value_iteration(self):
        self.value = np.zeros(self.num_states)
        # Start Iteration
        for i in range(self.iters):
            prev_value = self.value.copy()
            for state in range(self.num_states):
                action = self.policy[state]
                outcomes = []
                for prob, next_state, reward, done in self.env.P[state][action]:
                    outcomes.append(prob * (reward + self.gamma * prev_value[next_state]))
                self.value[state] = np.sum(outcomes)
            self.num_iters += 1
            if (np.sum(np.fabs(prev_value - self.value)) <= self.eps):
                return self.value
        
    def policy_iteration(self):
        self.num_iters = 0
        self.policy = np.zeros(self.num_states, dtype=int)
        for _ in range(self.iters):
            self.value_iteration()
            prev_Q = self.Q.copy()
            prev_policy = self.policy.copy()
            for state in range(self.num_states):
                Q = []
                for action in range(self.num_actions):
                    outcomes = []
                    for prob, next_state, reward, done in self.env.P[state][action]:
                        outcomes.append(prob * (reward + self.gamma * self.value[next_state]))
                    Q.append(np.sum(outcomes))
                self.Q[state] = Q
                self.policy[state] = np.argmax(Q)
#            print(self.policy, prev_policy)
            if np.array_equal(self.policy, prev_policy): #(np.sum(np.fabs(prev_Q - self.Q)) <= self.eps * self.num_actions):
                print(f"Converged after {self.num_iters} value iterations and {_+1} policy iterations")
                return self.value
        
    def run(self):
        startTime = time.time()
        self.policy_iteration()
        endTime = time.time()
        self.time_cost = endTime - startTime
        self.eval_policy()

class QLearningAgent(ValueIterationAgent):
    def __init__(self, env, gamma=1, iters=10000, eval_iters=100, eps=1e-20, seed=3547, lr=0.8):
        super().__init__(env, gamma, iters, eval_iters, eps, seed)
        self.lr = lr
        random.seed(seed)
 
    def q_learning(self):
        # Initialize Q table
        self.Q = np.zeros([self.num_states, self.num_actions])
        num_episodes = self.iters
        #create lists to contain total rewards and steps per episode
        rewards = []
        for i in range(num_episodes):
        #Reset environment and get first new observation
            s = self.env.reset()[0]
            #Total reward in one episode
            tot_reward = 0
            for _ in range(self.iters):
                # Choose an action by greedily (with noise) picking from Q table with given s
                noise_scale = max(1 / (i + 1), 0.002)
                action_evaluation = self.Q[s] + np.random.randn(1, self.num_actions) * noise_scale
                action = np.argmax(action_evaluation)
                # Get new state s1, reward and done from environment
                s1, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                # Update Q-Table with new knowledge
                self.Q[s, action] = (1 - self.lr) * self.Q[s, action] + self.lr * (reward + self.gamma * np.max(self.Q[s1]))
                # Cumulate the total reward
                tot_reward += reward
                # Update s
                s = s1
                if done == True:
                    break
            rewards.append(tot_reward)
        #print(self.Q)
        print("Score over time: " +  str(sum(rewards)/num_episodes))
        
    def get_policy(self):
        self.value = np.zeros(self.num_states)
        self.policy = np.zeros(self.num_states, dtype=int)
        for state in range(self.num_states):
            self.value[state] = np.max(self.Q[state])
#            print(self.value[state], self.Q[state])
            self.policy[state] = np.argmax(self.Q[state])

    def run(self):
        startTime = time.time()
        self.q_learning()
        self.get_policy()
        endTime = time.time()
        self.time_cost = endTime - startTime
        self.eval_policy()
