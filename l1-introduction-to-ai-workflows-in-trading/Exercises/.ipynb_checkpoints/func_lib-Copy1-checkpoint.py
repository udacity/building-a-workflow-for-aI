# Import yfinance library
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def createHistPrices():
    # Define the start and end dates  
    start_date = '2000-01-01'
    end_date   = '2024-05-01'
    
    # Define the list of tickers
    sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    
    # Filter out Class B shares that have a '.B' in the ticker name
    sp500_tickers = [ticker for ticker in sp500_tickers if '.B' not in ticker]
    
    # Download historical prices for the list of tickers
    historical_prices = yf.download(sp500_tickers, start=start_date, end=end_date)
    
    # Filter and keep only columns where the first level of the MultiIndex is 'Adj Close'
    historical_prices  = historical_prices.loc[:, historical_prices.columns.get_level_values(0) == 'Adj Close']
    
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

def compute_BM_Perf(total_returns):
    # Compute the daily mean of all stocks. This will be our equal weighted benchmark
    daily_mean  = pd.DataFrame(total_returns.loc[:,'F_1_d_returns'].groupby(level='Date').mean())
    daily_mean.rename(columns={'F_1_d_returns':'SP&500'}, inplace=True)
    
    # Convert daily returns to cumulative returns
    cum_returns = pd.DataFrame((daily_mean[['SP&500']]+1).cumprod())
    
    # Plotting the cumulative returns
    cum_returns.plot()
    
    # Customizing the plot
    plt.title('Cumulative Returns Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Return', fontsize=14)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend(title_fontsize='13', fontsize='11')
    
    # Display the plot
    plt.show()
    
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
    
    sharpe  = average_daily_return/stand_dev_dail_return
    
    print(f'Sharpe Ratio of Strategy: {round(sharpe.iloc[0],2)}')
    
    
    #df_daily_mean.rename(columns={target:'Strategy'},inplace=True)
    ann_returns = (pd.DataFrame((daily_mean[['SP&500']]+1).groupby(daily_mean.index.get_level_values(0).year).cumprod())-1)*100
    calendar_returns  = pd.DataFrame(ann_returns['SP&500'].groupby(daily_mean.index.get_level_values(0).year).last())
    
    calendar_returns.plot.bar(rot=30,  legend='top_left')#.opts(multi_level=False) 

    return cum_returns, calendar_returns



def compute_Strat_Perf(total_returns, cum_returns, calendar_returns, trading_strategy, model_name):    
    # Apply trading strategy to each RSI value
    total_returns['Position'] = total_returns[model_name].transform(trading_strategy)
    # Create Returns for each Trade
    total_returns[f'{model_name}_Return'] = total_returns['F_1_d_returns'] *  total_returns['Position'] 
    
    # Compute the daily mean of all stocks. This will be our equal weighted benchmark
    daily_mean  = pd.DataFrame(total_returns.loc[:,f'{model_name}_Return'].groupby(level='Date').mean())
    
    # Convert daily returns to cumulative returns
    cum_returns.loc[:,f'{model_name}_Return']  = pd.DataFrame((daily_mean[[f'{model_name}_Return']]+1).cumprod())

    # Plotting the cumulative returns
    cum_returns.plot()
    
    # Customizing the plot
    plt.title('Cumulative Returns Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Return', fontsize=14)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend(title_fontsize='13', fontsize='11')
    
    # Display the plot
    plt.show()
    
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
    
    ann_returns = (pd.DataFrame((daily_mean[f'{model_name}_Return']+1).groupby(daily_mean.index.get_level_values(0).year).cumprod())-1)*100
    
    
    calendar_returns.loc[:,f'{model_name}_Return']  = pd.DataFrame(ann_returns[f'{model_name}_Return'].groupby(daily_mean.index.get_level_values(0).year).last())
    
    calendar_returns.plot.bar(rot=30,  legend='top_left')#.opts(multi_level=False) 
    return cum_returns, calendar_returns