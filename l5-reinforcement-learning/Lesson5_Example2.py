
import random
import numpy as np

from collections import defaultdict

import gymnasium as gym
#from gymnasium import spaces
#from gymnasium import Env

class SimpleGridEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=5):
        super(SimpleGridEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = gym.spaces.Discrete(4)  # 4 actions: up, down, left, right
        self.observation_space = gym.spaces.MultiDiscrete([grid_size, grid_size])
        self.state = None
        self.goal = (grid_size - 1, grid_size - 1)
    
    def reset(self):
        self.state = (0, 0)
        return np.array(self.state, dtype=np.int32)
    
    def step(self, action):
        x, y = self.state
        
        if action == 0:  # up
            x = max(0, x - 1)
        elif action == 1:  # down
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # left
            y = max(0, y - 1)
        elif action == 3:  # right
            y = min(self.grid_size - 1, y + 1)
        
        self.state = (x, y)
        
        done = self.state == self.goal
        reward = 1 if done else -0.1
        
        return np.array(self.state, dtype=np.int32), reward, done, {}
    
    def render(self, mode='human'):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:] = '.'
        grid[self.goal] = 'G'
        x, y = self.state
        grid[x, y] = 'A'
        print("\n".join(["".join(row) for row in grid]))
        print()

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.env = env
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

def train_agent(env, agent, episodes=1000):
    for episode in range(episodes):
        state = tuple(env.reset())
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = tuple(next_state)
            agent.learn(state, action, reward, next_state)
            state = next_state
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1} completed")


if __name__ == "__main__":
    
    env = SimpleGridEnv()
    agent = QLearningAgent(env)
    train_agent(env, agent, episodes=1000)

    state = tuple(env.reset())
    done = False
    while not done:
        action = agent.choose_action(state)
        state, reward, done, _ = env.step(action)
        state = tuple(state)
        env.render()