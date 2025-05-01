# Value Iteration with Application on Tic-Tac-Toe (Zero-Sum) Game

## Introduction
This is a course project of DDA4300:Optimization in Data Science and Machine Learning, completed by [Yuan Xu](https://github.com/moonight3547/), [Wenda Lin](https://github.com/Linwd1), Jiaxing Wei, and Yanzhen Chen. Our first motivation is to finish the project requirements in `project_description.pdf`, wondering that we can discover how to implement value iteration methods on different environments and improve their performance. Furthermore, we study on the connection between Linear Programming in optimization, Markov Decision Process in Reinforcement Learning and Zero-sum Games in game theory, and want to verify that value iteration can solve the equivalent MDP of LP problems, while the zero-sum games in extensive form (EFG) can be flatten into a normal form game and the Nash equilibrium can be solved with an equivalent LP problem. 

About some details about implementation and experiments results, you can find them from `project_report.pdf`. 

## Environment Setup
You can follow the setup commands below with conda and pip:

```
conda create -n vi-zsg python=3.10
conda activate vi-zsg
pip install gym pygame matplotlib ipykernel
```

The GUI are implemented with pygame and turtle packages. 

## Implementation of Agents and Environments

The directory tree is shown below:
```
.
├── README.md
├── grid_world_env.py
├── project_description.pdf
├── project_report.pdf
├── random_value_iteration.png
├── random_value_iteration_time.png
├── tictactoe_2p_env.py
├── tictactoe_env.py
├── tictactoe_gui.py
├── tictactoe_gui_env.py
├── two_agents.ipynb
├── value_iteration.ipynb
├── value_iteration.py
└── zero_sum_env.py
```


### RL Agents: Value Iteration, Policy Iteration and Q-Learning

### RL Environments: Toy Environments and Zero-sum Games

## Experiments on Value Iteration Methods

We conduct a series of experiments for variants of value iteration methods on different environments in `value_iteration.ipynb`. 

### Update Ratio in Random Value Iteration (RVI)

We observe that the update ratio $\rho$ in Random Value Iteration (RVI) influence the performance of the methods on different environments. 
When $\rho$ is too small, the total rewards obtained from the environment will decrease. 
When $\rho$ is too large, Random Value Iteration cannot have an obvious speedup factor comparing with standard Value Iteration. 
Luckily, in the experiments, we find that these two marginal cases are not hard to avoid, which means we can often easily find a good parameter $\rho$ such that Random Value Iteration can perform very well (nearly reach the optimal rewards) with a great speedup factor ($\approx 10$). 

<img src="random_value_iteration.png" height="325"> <img src="random_value_iteration_time.png" height="325">

### Analysis of Cyclic Value Iteration (CVI)

## Applications on Zero-sum Games

### Nash Equilibrium of Normal Form Game (LP problem)

### Subgame Perfect Nash Equilibrium of Extensive Form Game (Tic-Tac-Toe)
We conduct an experiment on Tic-Tac-Toe game of two "optimal" agents in `two_agents.ipynb`. We find an optimal policy for both the first player and the second player, and they will reach a tie in the game of board size $3\times 3$. 
