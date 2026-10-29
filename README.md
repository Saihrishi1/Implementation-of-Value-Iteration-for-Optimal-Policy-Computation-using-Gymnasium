# Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium

---
## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman optimality equation.

---

## Problem Statement
To implement the Value Iteration algorithm for solving a finite Markov Decision Process (MDP) using the Gymnasium FrozenLake-v1 environment and determine the optimal state-value function and policy by iteratively applying the Bellman optimality equation until convergence.

## Software Requirements

- Python 3.10 or above
- Gymnasium
- NumPy
- Matplotlib
- Jupyter Notebook / Google Colab
- Operating System: Windows 10/11, Linux, or macOS

  
## Environment Description

The experiment uses the FrozenLake-v1 environment from the Gymnasium library. It is a 4×4 grid-world environment consisting of 16 discrete states and four possible actions: Left, Down, Right, and Up. The agent starts from the Start (S) state and aims to reach the Goal (G) while avoiding Holes (H). The Frozen (F) cells are safe to traverse. Since the surface is slippery, actions may not always produce the intended movement, making the environment stochastic and suitable for evaluating reinforcement learning algorithms.

## MDP Representation
The FrozenLake-v1 environment can be represented as a finite Markov Decision Process (MDP) defined by the tuple **(S, A, P, R, γ)**.

| Component | Description |
|-----------|-------------|
| **State Space (S)** | 16 discrete states corresponding to the cells of a 4×4 FrozenLake grid. |
| **Action Space (A)** | Four actions: **0 - Left**, **1 - Down**, **2 - Right**, **3 - Up**. |
| **Transition Probability (P)** | Specifies the probability of reaching the next state after taking an action. Due to the slippery surface, state transitions are stochastic. |
| **Reward Function (R)** | Reward = **1** upon reaching the Goal (G); Reward = **0** for all other transitions. |
| **Discount Factor (γ)** | A discount factor of **0.99** is used to maximize long-term cumulative rewards. |
| **Terminal States** | Goal (**G**) and Hole (**H**) states terminate the episode. |
| **Objective** | Compute the optimal state-value function and derive the optimal policy using the Value Iteration algorithm. |

## Theory

Value Iteration is a Dynamic Programming algorithm used to solve Markov Decision Processes (MDPs). It repeatedly updates the value of every state using the Bellman Optimality Equation until the values converge. The algorithm estimates the maximum expected cumulative reward that can be obtained from each state. Once convergence is achieved, the optimal policy is extracted by selecting the action that maximizes the expected future reward for every state. Value Iteration guarantees an optimal policy for finite MDPs when the discount factor is less than one.


## Algorithm

1. Initialize the value function V(s) of every state to zero.
2. Choose the discount factor (γ) and convergence threshold (θ).
3. For every state, compute the value of each possible action using the Bellman optimality equation.
4. Update the state value with the maximum action value.
5. Repeat the process until the maximum change in state values is less than the threshold.
6. After convergence, compute the optimal policy by selecting the action with the highest expected value for every state.
7. Display the optimal value function and policy.



## Python Program

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------
env_desc = [
    "HFGH",
    "FHFF",
    "FFFH",
    "SFFH"
]

env = gym.make("FrozenLake-v1", desc=env_desc, is_slippery=True)

n_states = env.observation_space.n
n_actions = env.action_space.n

# -------------------------------------------------
# Value Iteration Algorithm
# -------------------------------------------------

def value_iteration(env, gamma=0.99, theta=1e-8):
    """
    Performs value iteration and returns the optimal value function.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)
    iteration = 0

    while True:
        policy = np.zeros(n_states, dtype=int)
        delta = 0
        for state in range(n_states):
            action_values = []
            for action in range(n_actions):
                value = 0
                for probability, next_state, reward, terminated in env.unwrapped.P[state][action]:
                    value += probability * (reward + gamma * V[next_state] * (not terminated))
                action_values.append(value)
            policy[state] = int(np.argmax(action_values))

            best_value = max(action_values)
            delta = max(delta, abs(best_value - V[state]))
            V[state] = best_value

        iteration += 1
        if delta < theta:
            break      


    return V, policy, iteration


# -------------------------------------------------
# Run Value Iteration
# -------------------------------------------------
V, policy, iterations = value_iteration(env)



# -------------------------------------------------
# Display Output
# -------------------------------------------------
print("Name: Sai Hrishi M ")
print("Register Number: 212224240140 ")
print("Value Iteration Completed")
print("Number of Iterations:", iterations)

print("\nOptimal State-Value Function:")
print(np.round(V.reshape(4, 4), 4))


action_symbols = {
    0: "L",
    1: "D",
    2: "R",
    3: "U"
}

policy_grid = np.array(
    [action_symbols[action] for action in policy]
).reshape(4, 4)

print("\nOptimal Policy:")
print(policy_grid)

env.close()


```

---

## Output

Environment:
<br>
<br>
<img width="157" height="153" alt="image" src="https://github.com/user-attachments/assets/01f91942-f76c-4f51-be30-7d13d160dc6f" />
<br>
<br>
<img width="681" height="362" alt="image" src="https://github.com/user-attachments/assets/bb6bf875-243b-4be4-b656-25665d739a10" />

---

## Result
The Value Iteration algorithm successfully computed the optimal state-value function and generated the optimal policy for the FrozenLake-v1 environment after convergence.

---

## Inference

1. Convergence Performance: The Value Iteration algorithm successfully converged in 157 iterations for the custom layout, which is faster than the default layout due to fewer obstruction bottlenecks.
2. State-Value Distribution: States closest to the goal (`G`) exhibit the highest expected cumulative values (peaking around 0.9522), while terminal/hole states drop to 0.
3. Optimal Policy Mapping: The resulting policy correctly maps directional actions (`U`, `D`, `L`, `R`) to guide the agent safely along the optimal path toward the target.
---



