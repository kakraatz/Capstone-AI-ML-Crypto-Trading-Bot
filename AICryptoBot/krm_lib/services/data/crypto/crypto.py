
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 09:46:26 2023

@author: JohnMurphy
"""

# Data Preprocessing
import pandas as pd
import numpy as np
from krm_lib.services.apis.binance import BinanceAPI
#from pandas_datareader import data as pdr
#import yfinance as yfin
# from ta.volume import VolumeWeightedAveragePrice
from ta import add_all_ta_features

# Statistics
from statsmodels.tsa.stattools import adfuller

# Unsupervised Machine Learning
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

class CryptoHistory():
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start = start_date
        self.end = end_date
        
    def get_scaled_price_df(self):
        binance = BinanceAPI()
        df = binance.get_daily_price_history(self.symbol, self.start, self.end, 'dataframe')
        df.set_index("Open_Time", inplace = True)
        df.drop(columns=["Close_Time", "Quote_Asset_Volume", "Taker_Buy_Base_Volume", "Taker_Buy_Quote_Volume", "Ignore"
                         ], inplace=True)
        df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
#        vwap = VolumeWeightedAveragePrice(high=df["High"], low=df["Low"], close=df["Close"],
#                                         volume=df["Volume"], window=14, fillna=False)
#       df["VWAP"] = vwap.volume_weighted_average_price()

        # Identify non-stationary columns
        non_stationaries = []
        for col in df.columns:
            dftest = adfuller(df[col].values)
            p_value = dftest[1]
            t_test = dftest[0] < dftest[4]["1%"]
            if p_value < 0.05 or not t_test:
                non_stationaries.append(col)

#       print(f"Non-stationaries Features Found: {len(non_stationaries)}")
#       print(non_stationaries)
#       df.dropna(inplace=True)

        # Convert non-stationaries to stationaries
        df_stationary = df.copy()
        df_stationary[non_stationaries] = df_stationary[non_stationaries].pct_change()
        df_stationary = df_stationary.iloc[1:]

        # Find NaN Rows
        na_list = df_stationary.columns[df_stationary.isna().any().tolist()]
        df_stationary.drop(columns=na_list, inplace=True)

        # Handle inf values
        df_stationary.replace([np.inf, -np.inf], 0, inplace=True)

        # Min Max Scaled
        df_mod = df_stationary.copy()
#        print(df_mod)
#        df_mod = df_mod.pct_change() * 100
        df_mod = df_mod / df_mod.max()  # Problem line, need to make every value an integer
#        print(df_mod)
#        df_mod = df_mod.dropna()
        df_mod = df_mod.reset_index(drop=True)
        df_mod["Close_Price"] = df["Close"].iloc[1:].values

        # Splitting the data
        X = df_mod.iloc[:, :-1]
        y = df_mod.iloc[:, -1]

        # PCA
        n_components = 5
        pca = PCA(n_components=n_components)
        pca.fit(X)
        X_pca = pca.transform(X)

        # Calculate the variance explained by Principle Components
        print("Variance of each component: ", pca.explained_variance_ratio_)
        print("\n Total variance explained: ", round(sum(list(pca.explained_variance_ratio_)) * 100, 2))

        pca_cols = []
        for i in range(n_components):
            pca_cols.append(f"PC_{i}")

        # # Create and View Dataframe
        df_pca = pd.DataFrame(data=X_pca, columns=pca_cols)
        df_pca["Close_Price"] = y.iloc[:].values
        print(df_pca)
        return df_pca
    
    def get_price_data(self):
        binance = BinanceAPI()
        df = binance.get_daily_price_history(self.symbol, self.start, self.end, 'dataframe')
        return df

 #   def get_pca_indicators(self):

