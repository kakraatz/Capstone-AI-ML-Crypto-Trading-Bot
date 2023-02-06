# -*- coding: utf-8 -*-
"""
John Murphy, Kevin Kraatz, Richard Silva
25 Jan 2023
ML Sin Wave Trading Bot
"""

# Enviornment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Create timesteps
time = np.arange(0, 50, 0.1)

# Assign amplitude and normalise above 0
amplitude = np.sin(time)
amplitude = amplitude + 1
max_amp = max(amplitude)
amplitude = amplitude / max_amp

# Construct DataFrame
df = pd.DataFrame(amplitude)
df.columns = ["Close"]
df["Close_Rt"] = df["Close"].pct_change()
df = df.replace(np.inf, np.nan)
df = df.dropna()
df = df.reset_index(drop=True)

# Show DataFrame and Values
print(f"length : {len(df)}")
print("Min Close: ", df["Close"].min())
print("Max Close: ", df["Close"].max())
df.head()

##################################
## Step 2 - Enviornment Setup and Class
## 
''''''
import gym
from gym import spaces 
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# Initialise variables
MAX_INT = 2147483647
MAX_OPEN_POSITIONS = 1
INITIAL_ACCOUNT_BALANCE = 1000
PERCENT_CAPITAL = 0.1
TRADING_COSTS_RATE = 0.001
KILL_THRESH = 0.4 # Threshold for balance preservation

