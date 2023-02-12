# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 09:16:50 2023

@author: JohnMurphy
"""
from krm_lib.services.machinelearning.rl.enviornments.trading import TradingEnv
from krm_lib.services.machinelearning.rl.agents.ppo_agent import Agent
from krm_lib.services.data.stock.stocks import StockHistory
from krm_lib.services.apis.binance import BinanceAPI
import matplotlib.pyplot as plt
import pandas as pd
import datetime 
import json

# Data Extraction
'''
START_DATE = "2017-01-1"
END_DATE = "2022-06-01"
SYMBOL = "AAPL"
'''

# Data Extraction
START_DATE = "2020-01-01"
END_DATE = "2022-12-31"
SYMBOL = "BTCUSD"




alpha = 0.0005
if __name__ == '__main__':
    # STOCK TRADING - START
    '''
    stock = StockHistory(symbol=SYMBOL, start_date=START_DATE, end_date=END_DATE)
    df = stock.get_scaled_price_df()
    # Split Training and Testing
    df_train = df.copy()
    df_train = df_train.iloc[:700]
    df_test = df.copy()
    df_test = df_test.iloc[700:]

    # View price behaviour
    plt.rcParams["figure.figsize"] = (15,5)
    df_train["Close_Price"].plot()
    df_test["Close_Price"].plot()
    
    '''
    # TRADING DATAFRAME - END
    
    # STOCK TRADING - END
    
    # CRYPTO TRADING - START
    binance = BinanceAPI()
    json_obj = binance.get_server_time()
    pdf = binance.get_daily_price_history(SYMBOL, START_DATE, END_DATE, 'dataframe')
    #json_obj = binance.get_daily_price_history(SYMBOL, START_DATE, END_DATE, 'json')
    server_time = datetime.datetime.fromtimestamp(json_obj["serverTime"]/1000)
    
    #json_formatted_str = json.dumps(json_obj, indent=4)
    #print(json_formatted_str)
    # CRYPTO TRADING - END
    