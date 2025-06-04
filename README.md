# EX-NO 03 : POLICY ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the policy iteration algorithm.

## PROBLEM STATEMENT
The FrozenLake environment is a benchmark reinforcement learning task where an agent must navigate a grid-based frozen surface to reach a goal state without falling into holes. The environment is stochastic, meaning that the agent's actions have uncertain outcomes due to slippery tiles.

The environment is modeled as a Markov Decision Process (MDP) with:

- States: Each cell in the grid.

- Actions: Move left, right, up, or down.

- Transition Probabilities: Due to the slippery nature, intended actions might not always be executed.

- Rewards: +1 for reaching the goal (G), 0 otherwise.

The goal is to determine the optimal policy—a mapping from states to actions—that maximizes the cumulative reward. This is done using the Policy Iteration algorithm, which iteratively evaluates and improves a policy until convergence.

## POLICY ITERATION ALGORITHM

![image](https://github.com/user-attachments/assets/7209c4fa-d76c-4a73-a376-2dfd1f01297f)

## POLICY IMPROVEMENT FUNCTION

```PY
DEVELOPED BY : SHALINI K
REGISTER NUMBER : 212222240095

def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    # Write your code here to improve the given policy
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob,next_state,reward,done in P[s][a]:
          Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
          new_pi=lambda s:{s:a for s, a in enumerate(np.argmax(Q,axis=1))}[s]
    return new_pi

```
## POLICY ITERATION FUNCTION

```PY
DEVELOPED BY : SHALINI K
REGISTER NUMBER : 212222240095

def policy_iteration(P, gamma=1.0, theta=1e-10):
    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    pi = lambda s: {s: a for s, a in enumerate(random_actions)}[s]

    while True:
        old_pi = {s: pi(s) for s in range(len(P))}
        V = policy_evaluation(pi, P, gamma, theta)
        pi = policy_improvement(V, P, gamma)

        if old_pi == {s: pi(s) for s in range(len(P))}:
            break

    return V, pi

```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy

![image](https://github.com/user-attachments/assets/420a36cf-cac9-43a9-92cf-976211d24092)
![image](https://github.com/user-attachments/assets/fcfd538e-6523-4209-af68-455dccf74735)
![image](https://github.com/user-attachments/assets/408c1231-045b-4868-8b55-2e0d5a48e5c1)

### 2. Policy, Value function and success rate for the Improved Policy

![image](https://github.com/user-attachments/assets/420a36cf-cac9-43a9-92cf-976211d24092)
![image](https://github.com/user-attachments/assets/fcfd538e-6523-4209-af68-455dccf74735)
![image](https://github.com/user-attachments/assets/408c1231-045b-4868-8b55-2e0d5a48e5c1)

### 3. Policy, Value function and success rate after policy iteration

![image](https://github.com/user-attachments/assets/3939a628-d659-429e-92cc-fdc2713c44d0)
![image](https://github.com/user-attachments/assets/a12c6f2c-b1b3-4131-87f5-62ba05926db4)
![image](https://github.com/user-attachments/assets/31e144df-9a0c-4302-9f6a-c23713ca1979)


## RESULT:
Thus, The Python program to find the optimal policy for the given MDP using the policy iteration algorithm is successfully executed.
