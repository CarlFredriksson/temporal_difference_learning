import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Environment:
    def __init__(self, starting_state):
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.goal = (7, 3)
        self.state = starting_state

    def adjust_position(self):
        # X position
        if self.state[0] < 0:
            self.state = (0, self.state[1])
        elif self.state[0] > 9:
            self.state = (9, self.state[1])

        # Y Position
        if self.state[1] < 0:
            self.state = (self.state[0], 0)
        elif self.state[1] > 6:
            self.state = (self.state[0], 6)

    def take_action(self, action):
        # Move agent from action
        if action == Action.UP:
            self.state = (self.state[0], self.state[1] + 1)
        elif action == Action.DOWN:
            self.state = (self.state[0], self.state[1] - 1)
        elif action == Action.LEFT:
            self.state = (self.state[0] - 1, self.state[1])
        elif action == Action.RIGHT:
            self.state = (self.state[0] + 1, self.state[1])
        else:
            print("ERROR: INVALID ACTION SELECTED!")
        self.adjust_position()

        # Move agent from wind
        if self.state != self.goal:
            self.state = (self.state[0], self.state[1] + self.wind[self.state[0]])
            self.adjust_position()

        if self.state == self.goal:
            return 0
        return -1

class SarsaAgent:
    def __init__(self, alpha, epsilon):
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = np.zeros((10, 7, 4))

    def take_step(self, environment):
        s = environment.state

        # Select action epsilon-greedily
        a = np.argmax(self.Q[s[0], s[1], :])
        if np.random.rand() < self.epsilon:
            a = np.random.randint(0, 4)
        r = environment.take_action(Action(a))

        # Update Q
        s_prime = environment.state
        a_prime = np.argmax(self.Q[s_prime[0], s_prime[1], :])
        if np.random.rand() < self.epsilon:
            a_prime = np.random.randint(0, 4)
        self.Q[s[0], s[1], a] += self.alpha * (r + self.Q[s_prime[0], s_prime[1], a_prime] - self.Q[s[0], s[1], a])

        # Check if goal has been reached
        if r == 0:
            return True
        return False

class QLearningAgent:
    def __init__(self, alpha, epsilon):
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = np.zeros((10, 7, 4))

    def take_step(self, environment):
        s = environment.state

        # Select action epsilon-greedily
        a = np.argmax(self.Q[s[0], s[1], :])
        if np.random.rand() < self.epsilon:
            a = np.random.randint(0, 4)
        r = environment.take_action(Action(a))

        # Update Q
        s_prime = environment.state
        self.Q[s[0], s[1], a] += self.alpha * (r + np.max(self.Q[s_prime[0], s_prime[1], :]) - self.Q[s[0], s[1], a])

        # Check if goal has been reached
        if r == 0:
            return True
        return False

if __name__ == "__main__":
    NUM_EPISODES = 200
    MAX_NUM_STEPS = 1000
    ALPHA = 0.5
    EPSILON = 0.1
    STARTING_STATE = (0, 3)

    agents = [
        SarsaAgent(ALPHA, EPSILON),
        QLearningAgent(ALPHA, EPSILON)
    ]
    num_steps = np.zeros((len(agents), NUM_EPISODES))

    # Agents learning
    for agent_index, agent in enumerate(agents):
        for episode in range(NUM_EPISODES):
            environment = Environment(STARTING_STATE)
            for num_steps[agent_index, episode] in range(MAX_NUM_STEPS):
                at_goal = agent.take_step(environment)
                if at_goal:
                    break

    # Plot results
    plt.plot(range(NUM_EPISODES), num_steps[0, :], label="Sarsa")
    plt.plot(range(NUM_EPISODES), num_steps[1, :], label="Q-learning")
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Number of steps to reach goal")
    plt.show()
