# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 09:46:26 2023

@author: JohnMurphy
"""

# Data Preprocessing
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yfin
from ta.volume import VolumeWeightedAveragePrice

class StockHistory():
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start = start_date
        self.end = end_date
        yfin.pdr_override()
        
    def get_scaled_price_df(self):
        df = pdr.get_data_yahoo(self.symbol, start=self.start, end=self.end)
        df.drop(columns=["Adj Close"], inplace=True)
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
        # Split Training and Testing
        '''
        df_train = df_mod.copy()
        df_train = df_train.iloc[:700]
        df_test = df_mod.copy()
        df_test = df_test.iloc[700:]
        '''
        
        