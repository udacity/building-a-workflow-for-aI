import random
import numpy as np
import pandas as pd
from collections import defaultdict
import gymnasium as gym
from gymnasium import spaces

# Define the custom gym environment
class ReturnEnv(gym.Env):
    def __init__(self, df):
        super(ReturnEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0 or 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        
        # Initialize dataframe
        self.df = df
        self.current_step = 0
        
    def reset(self):
        self.current_step = 0
        state = self.df.iloc[self.current_step][['1_d_returns', '2_d_returns']].values
        return state
    
    def step(self, action):
        target = self.df.iloc[self.current_step]['Target_Returns']
        
        # Reward if action matches target return
        reward = 1 if action == target else -1
        
        self.current_step += 1
        done = self.current_step >= len(self.df)
        
        if not done:
            next_state = self.df.iloc[self.current_step][['1_d_returns', '2_d_returns']].values
        else:
            next_state = np.zeros(2)
        
        return next_state, reward, done, {}

# Q-learning Agent
class QLearningAgent:
    def __init__(self, action_space, state_space, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.action_space = action_space
        self.state_space = state_space
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))
    
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()  # explore
        else:
            return np.argmax(self.q_table[str(state)])  # exploit
    
    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[str(next_state)])
        td_target = reward + self.gamma * self.q_table[str(next_state)][best_next_action]
        td_error = td_target - self.q_table[str(state)][action]
        self.q_table[str(state)][action] += self.alpha * td_error

def main():
    # Example data
    df = pd.DataFrame({
        'Target_Returns': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        '1_d_returns': [0.062030, -0.038076, 0.050, 0.030, -0.020, 0.062030, -0.038076, 0.050, 0.030, -0.020],
        '2_d_returns': [0.133681, -0.097744, 0.070, 0.040, -0.010, 0.133681, -0.097744, 0.070, 0.040, -0.010]
    })


    # Create environment and agent
    env = ReturnEnv(df)
    agent = QLearningAgent(env.action_space, env.observation_space)

    # Training loop
    n_episodes = 1000
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state

    print("Training finished.")

    # Evaluation
    state = env.reset()
    done = False
    step = 1

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        
        # Print detailed step information
        print(f"Action: {action}, Target Return: {env.df['Target_Returns'].values[step-1]}, Reward: {reward}, Step: {step}, State: {state}, Next State: {next_state}")
        
        state = next_state
        step += 1

if __name__ == "__main__":
    main()
