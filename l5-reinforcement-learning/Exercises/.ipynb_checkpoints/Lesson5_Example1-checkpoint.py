import gymnasium as gym
from stable_baselines3 import DQN
import time

# Create the environment with render mode set to 'human'
env = gym.make('CartPole-v1', render_mode='human')

# Create the agent with verbose turned off
agent = DQN('MlpPolicy', env, verbose=0)

# Train the agent
agent.learn(total_timesteps=10000)

# Test the trained agent
obs, _ = env.reset()
done = False
while not done:
    action, _ = agent.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    env.render()
    time.sleep(0.05)  # Add a small delay to see the rendering

env.close()