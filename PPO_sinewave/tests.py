# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 13:23:02 2023

@author: JohnMurphy
"""

# Data Preprocessing
import pandas as pd
#from pandas_datareader.data import DataReader
from pandas_datareader import data as pdr
import yfinance as yfin
from ta.volume import VolumeWeightedAveragePrice

# Environment
import gym
from gym import spaces
import numpy as np
import random
import torch

# PyTorch
import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# Outputs
import matplotlib.pyplot as plt

# Y-Finance override()
yfin.pdr_override()
# Data Extraction
start_date = "2017-01-1"
end_date = "2022-06-01"
symbol = "AAPL"

df = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
#df = DataReader(name=symbol, data_source='yahoo', start=start_date, end=end_date)
df.drop(columns=["Adj Close"], inplace=True)
df.head(2)