import random
import numpy as np
import pandas as pd
from collections import defaultdict
import gymnasium as gym
from gymnasium import spaces

# Import yfinance library
import yfinance as yf
import pandas as pd
import numpy as np
#import hvplot.pandas
import matplotlib.pyplot as plt
pd.set_option('display.float_format', lambda x: '%.4f' % x)

import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Create a function called 'computingReturns' that takes prices and a list of integers (momentums) as an inputs
def computingReturns(prices, list_of_momentums): 
    # Initialize the forecast horizon
    forecast_horizon = 1
    # Compute forward returns by taking percentage change of close prices
    # and shifting by the forecast horizon
    f_returns = prices.pct_change(forecast_horizon, fill_method=None)
    f_returns = f_returns.shift(-forecast_horizon)
    # Convert the result to a DataFrame
    f_returns = pd.DataFrame(f_returns.unstack())
    # Name the column based on the forecast horizon
    name = "F_" + str(forecast_horizon) + "_d_returns"
    f_returns.rename(columns={0: name}, inplace=True)
    # Initialize total_returns with forward returns
    total_returns = f_returns
    
    # Iterate over the list of momentum values
    for i in list_of_momentums:   
        # Compute returns for each momentum value
        feature = prices.pct_change(i, fill_method=None)
        feature = pd.DataFrame(feature.unstack())
        # Name the column based on the momentum value
        name = str(i) + "_d_returns"        
        feature.rename(columns={0: name}, inplace=True)
        # Rename columns and reset index
        feature.rename(columns={0: name, 'level_0': 'Ticker'}, inplace=True)
        # Merge computed feature returns with total_returns based on Ticker and Date
        total_returns = pd.merge(total_returns, feature, left_index=True, right_index=True,how='outer')
    
    # Drop rows with any NaN values
    total_returns.dropna(axis=0, how='any', inplace=True) 

    # Return the computed total returns DataFrame
    return total_returns


def createHistoricalData():
    # Define the start and end dates  
    start_date = '2000-01-01'
    end_date   = '2024-05-01'

    # Define the list of tickers
    sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()

    # Filter out Class B shares that have a '.B' in the ticker name
    sp500_tickers = [ticker for ticker in sp500_tickers if '.B' not in ticker]

    # Download historical prices for the list of tickers
    historical_prices = yf.download(sp500_tickers, start=start_date, end=end_date)

    # Remove the MultiIndex and keep only the second level
    historical_prices.columns = historical_prices.columns.droplevel(0)   

    MIN_REQUIRED_NUM_OBS_PER_TICKER = 100
    # Count non-missing values for each ticker
    ticker_counts = historical_prices.count()

    # Filter out tickers with fewer than n=MIN_REQUIRED_NUM_OBS_PER_TICKER=100 non-missing values
    valid_tickers_mask = ticker_counts[ticker_counts >= MIN_REQUIRED_NUM_OBS_PER_TICKER].index

    # Filter the DataFrame based on valid tickers
    historical_prices = historical_prices[valid_tickers_mask]

    return historical_prices



def createHistoricalReturns(historical_prices):
    # Subsetting the data
    historical_prices = historical_prices[historical_prices.index.get_level_values('Date') > '2024-04-31']
    # Create a list of momentums for 1d and 2d returns
    list_of_momentums = [1, 5, 15, 20]
    # Run the function computingReturns and save the output as total_data
    total_returns = computingReturns(historical_prices, list_of_momentums)

    # Converting the 'F_1_d_returns' to binary based on whether the value is positive or not
    total_returns['F_1_d_returns_Ind'] = total_returns['F_1_d_returns'].apply(lambda x: 1 if x > 0 else 0)

    # Determine the split index for 70% of the dates
    unique_dates = total_returns.index.get_level_values('Date').unique()
    split_date = unique_dates[int(0.7 * len(unique_dates))]

    # Create the training set: all data before the split date
    train_data = total_returns.loc[total_returns.index.get_level_values('Date') < split_date]

    # Create the testing set: all data from the split date onwards
    test_data = total_returns.loc[total_returns.index.get_level_values('Date') >= split_date]

    features = ['1_d_returns', '5_d_returns', '15_d_returns', '20_d_returns']
    target   = ['F_1_d_returns']

    total_returns  = test_data[target]
    target   = ['F_1_d_returns_Ind']

    # Split the data into training and testing sets
    X_train = train_data[features]
    X_test  = test_data[features]
    y_train = train_data[target]
    y_test  = test_data[target]

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled  = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

    # Compute the daily mean of all stocks. This will be our equal weighted benchmark
    daily_mean  = pd.DataFrame(total_returns.loc[:,'F_1_d_returns'].groupby(level='Date').mean())
    daily_mean.rename(columns={'F_1_d_returns':'SP&500'}, inplace=True)

    # Convert daily returns to cumulative returns
    cum_returns = pd.DataFrame((daily_mean[['SP&500']]+1).cumprod())


    # # Plotting the cumulative returns
    # cum_returns.plot()

    # # Customizing the plot
    # plt.title('Cumulative Returns Over Time', fontsize=16, fontweight='bold')
    # plt.xlabel('Date', fontsize=14)
    # plt.ylabel('Cumulative Return', fontsize=14)
    # plt.grid(True)
    # plt.xticks(rotation=45)
    # plt.legend(title_fontsize='13', fontsize='11')

    # # Display the plot
    # plt.show()

    # Calculate the number of years in the dataset
    number_of_years = len(daily_mean) / 252  # Assuming 252 trading days in a year

    ending_value    = cum_returns['SP&500'].iloc[-1]
    beginning_value = cum_returns['SP&500'].iloc[1]

    # Compute the Compound Annual Growth Rate (CAGR)
    ratio = ending_value/beginning_value
    cagr = round((ratio**(1/number_of_years)-1)*100,2)
    print(f'The CAGR is: {cagr}%')

    # Compute the Sharpe Ratio by annualizing the daily mean and the daily std
    average_daily_return    = daily_mean[['SP&500']].describe().iloc[1,:] * 252
    stand_dev_dail_return   = daily_mean[['SP&500']].describe().iloc[2,:] * pow(252,1/2)

    # Compute the Sharpe Ratio by annualizing the daily mean and the daily std
    average_daily_return    = daily_mean[['SP&500']].describe().iloc[1,:] * 252
    stand_dev_dail_return   = daily_mean[['SP&500']].describe().iloc[2,:] * pow(252,1/2)

    sharpe  = average_daily_return/stand_dev_dail_return

    print(f'Sharpe Ratio of Strategy: {round(sharpe.iloc[0],2)}')

    #df_daily_mean.rename(columns={target:'Strategy'},inplace=True)
    ann_returns = (pd.DataFrame((daily_mean[['SP&500']]+1).groupby(daily_mean.index.get_level_values(0).year).cumprod())-1)*100
    calendar_returns  = pd.DataFrame(ann_returns['SP&500'].groupby(daily_mean.index.get_level_values(0).year).last())

    #calendar_returns.hvplot.bar(rot=30,  legend='top_left').opts(multi_level=False) 

    return calendar_returns, ann_returns, X_train_scaled, X_test_scaled, y_train, y_test, total_returns




# Define the custom gym environment
class ReturnEnv(gym.Env):
    def __init__(self, df):
        super(ReturnEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0 or 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        # Initialize dataframe
        self.df = df.reset_index()
        self.current_step = 0
        
    def reset(self):
        self.current_step = 0
        state = self.df.iloc[self.current_step][['1_d_returns', '5_d_returns', '15_d_returns', '20_d_returns']].values
        return state
    
    def step(self, action):
        target = self.df.iloc[self.current_step]['F_1_d_returns_Ind']
        
        # Reward if action matches target return
        reward = 1 if action == target else -1
        
        self.current_step += 1
        done = self.current_step >= len(self.df)
        
        if not done:
            next_state = self.df.iloc[self.current_step][['1_d_returns', '5_d_returns', '15_d_returns', '20_d_returns']].values
        else:
            next_state = np.zeros(4)
        
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

def print_predictions(env, agent, df, dataset_name):
    state = env.reset()
    done = False
    step = 1
    predictions = []

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        
        # Record prediction details
        predictions.append((env.current_step - 1, action, reward, state, next_state))

        state = next_state
        step += 1
    
    # Print predictions
    print(f"\nPredictions for {dataset_name} data:")
    for idx, action, reward, state, next_state in predictions:
        print(f"Action: {action}, Target Return: {env.df.loc[idx]['F_1_d_returns_Ind']}, Reward: {reward}, Step: {step}, State: {state}, Next State: {next_state}")

    # Add predictions to DataFrame
    pred_df = pd.DataFrame(predictions, columns=['Index', 'Action', 'Reward', 'State', 'Next_State'])
    pred_df = pred_df.set_index('Index')
    df = df.reset_index().merge(pred_df[['Action']], left_index=True, right_index=True, how='left').set_index(df.index.names)
    return df

def trading_strategy(y_pred):
    if y_pred >0.5:
        return  1 # Go long
    else:
        return  0  
    
def main():
    # Training dataset

    # # historical_prices = createHistoricalData()
    historical_prices = pd.read_csv('historical_prices.csv')
    historical_prices['Date'] = pd.to_datetime(historical_prices['Date'])
    historical_prices.set_index('Date', inplace=True)
    calendar_returns, ann_returns, X_train_scaled, X_test_scaled, y_train, y_test, total_returns = createHistoricalReturns(historical_prices)
    df  = pd.merge(y_train, X_train_scaled, left_index=True, right_index=True)
    df2 = pd.merge(y_test,  X_test_scaled,  left_index=True, right_index=True)

    # df = pd.DataFrame({
    #     '1_d_returns': [0.0727, 0.0775, 0.0163, -0.0193, 0.0426],
    #     '5_d_returns': [0.0489, 0.1199, 0.1361, 0.1203, 0.2011],
    #     '15_d_returns': [0.0299, 0.1250, 0.1670, 0.1275, 0.1627],
    #     '20_d_returns': [-0.0139, 0.1504, 0.2465, 0.2708, 0.2231],
    #     'F_1_d_returns_Ind': [1, 1, 0, 1, 0]
    # }, index=pd.MultiIndex.from_tuples([
    #     ('A', '2000-02-01'),
    #     ('A', '2000-02-02'),
    #     ('A', '2000-02-03'),
    #     ('A', '2000-02-04'),
    #     ('A', '2000-02-07')
    # ], names=['Ticker', 'Date']))

    # # Evaluation dataset
    # df2 = pd.DataFrame({
    #     '1_d_returns': [0.0334, 0.0589, 0.0110, -0.0123, 0.0321],
    #     '5_d_returns': [0.0450, 0.1099, 0.1261, 0.1103, 0.2015],
    #     '15_d_returns': [0.0239, 0.1150, 0.1570, 0.1175, 0.1527],
    #     '20_d_returns': [-0.0135, 0.1404, 0.2365, 0.2608, 0.2131],
    #     'F_1_d_returns_Ind': [1, 0, 1, 0, 1]
    # }, index=pd.MultiIndex.from_tuples([
    #     ('A', '2000-02-08'),
    #     ('A', '2000-02-09'),
    #     ('A', '2000-02-10'),
    #     ('A', '2000-02-11'),
    #     ('A', '2000-02-14')
    # ], names=['Ticker', 'Date']))

    # Create environment and agent for training
    env = ReturnEnv(df)
    agent = QLearningAgent(env.action_space, env.observation_space)

    # Training loop
    n_episodes = 1
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state

    print("Training finished.")

    # Print predictions for training data and add them back to df
    df = print_predictions(env, agent, df, "training")

    # Create environment for evaluation
    eval_env = ReturnEnv(df2)

    # Print predictions for evaluation data and add them back to df2
    df2 = print_predictions(eval_env, agent, df2, "evaluation")

    #print("\nTraining Data with Predictions:")
    #print(df)
    
    #print("\nEvaluation Data with Predictions:")
    #print(df2)

    y_test_and_pred = pd.merge(df2, total_returns, left_index=True, right_index=True)
    model_name = 'RL'
    # Define trading strategy based on RSI

            
    # Apply trading strategy to each RSI value
    y_test_and_pred['Position'] = y_test_and_pred['y_pred'].transform(trading_strategy)
    # Create Returns for each Trade
    y_test_and_pred[f'{model_name}_Return'] = y_test_and_pred['F_1_d_returns'] *  y_test_and_pred['Position'] 

    # Compute the daily mean of all stocks. This will be our equal weighted benchmark
    daily_mean  = pd.DataFrame(y_test_and_pred.loc[:,f'{model_name}_Return'].groupby(level='Date').mean())

    # Convert daily returns to cumulative returns
    cum_returns.loc[:,f'{model_name}_Return'] = pd.DataFrame((daily_mean[[f'{model_name}_Return']]+1).cumprod())

    # Calculate the number of years in the dataset
    number_of_years = len(daily_mean) / 252  # Assuming 252 trading days in a year

    ending_value    = cum_returns[f'{model_name}_Return'].iloc[-1]
    beginning_value = cum_returns[f'{model_name}_Return'].iloc[1]

    ratio = ending_value/beginning_value
    # Compute the Compound Annual Growth Rate (CAGR)
    cagr = round((ratio**(1/number_of_years)-1)*100,2)

    print(f'The CAGR is: {cagr}%')

    # Compute the Sharpe Ratio by annualizing the daily mean and the daily std
    average_daily_return  = daily_mean[[f'{model_name}_Return']].describe().iloc[1,:] * 252
    stand_dev_dail_return   = daily_mean[[f'{model_name}_Return']].describe().iloc[2,:] * pow(252,1/2)

    # Compute the Sharpe Ratio and print it out
    sharpe  = average_daily_return/stand_dev_dail_return

    print(f'Sharpe Ratio of Strategy: {round(sharpe.iloc[0],2)}')
if __name__ == "__main__":
    main()


