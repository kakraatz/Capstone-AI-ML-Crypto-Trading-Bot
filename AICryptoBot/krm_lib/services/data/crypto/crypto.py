
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 09:46:26 2023

@author: JohnMurphy
"""

# Data Preprocessing
import pandas as pd
from krm_lib.services.apis.binance import BinanceAPI
#from pandas_datareader import data as pdr
#import yfinance as yfin
from ta.volume import VolumeWeightedAveragePrice

class CryptoHistory():
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start = start_date
        self.end = end_date
        
    def get_scaled_price_df(self):
        binance = BinanceAPI()
        df = binance.get_daily_price_history(self.symbol, self.start, self.end, 'dataframe')
        df.set_index("Open_Time", inplace = True)
        df.drop(columns=["Close_Time", "Quote_Asset_Volume", "Taker_Buy_Base_Volume", "Taker_Buy_Quote_Volume", "Ignore"], inplace=True)
        vwap = VolumeWeightedAveragePrice(high=df["High"], low=df["Low"], close=df["Close"], 
                                          volume=df["Volume"], window=14, fillna=False)
        df["VWAP"] = vwap.volume_weighted_average_price()
        df.dropna(inplace=True)
        # Min Max Scaled
        df_mod = df.copy()
        df_mod = df_mod.pct_change() * 100
        df_mod = df_mod / df_mod.max()
        df_mod = df_mod.dropna()
        df_mod = df_mod.reset_index(drop=True)
        df_mod["Close_Price"] = df["Close"].iloc[1:].values
        return df_mod
    
    def get_price_data(self):
        binance = BinanceAPI()
        df = binance.get_daily_price_history(self.symbol, self.start, self.end, 'dataframe')
        return df
    