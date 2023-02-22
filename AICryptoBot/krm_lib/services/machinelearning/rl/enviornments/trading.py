# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 08:53:35 2023

@author: JohnMurphy
"""
# Environment
import gym
from gym import spaces
import numpy as np
import random
import torch

# ENVIORNMENT START
# Initialise variables
MAX_INT = 2147483647
MAX_TRADES = 10000
MAX_OPEN_POSITIONS = 1
PERCENT_CAPITAL = 0.1
KILL_THRESH = 0.4 # Threshold for balance preservation

# Structure environment
class TradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_account_balance=1000, trading_cost_rate=0.001):
        super(TradingEnv, self).__init__()
        
        # Generic variables
        self.df = df
        
        # Account variables
        self.initial_balance = initial_account_balance
        self.available_balance = initial_account_balance
        self.trading_cost_rate = trading_cost_rate
        self.net_profit = 0
        
        # Position variables
        self.num_trades_long = 0
        self.num_trades_short = 0
        self.long_short_ratio = 0
        
        # Current Step
        self.current_step = 0
        self.lag = 20
        #self.lag = 10
        self.volatility = 1
        self.max_steps = len(df)

        # Actions of the format Long, Hold, Close
        self.action_space = spaces.Discrete(2)

        # Prices contains the Close and Close Returns etc
        self.observation_space = spaces.Box(low=-1, high=1, shape=(7, ), dtype=np.float32)

    # Calculate Reward
    def _calculate_reward(self):
        reward = 0
        reward += self.net_profit / self.volatility
        reward += 0.01 if self.long_short_ratio >= 0.3 and self.long_short_ratio <= 0.6 else -0.01
        return reward
        
    # Structure sign observation data
    def _next_observation(self):
        
        # item_0_T0 = self.df.loc[self.current_step - 0, "Open"].item()
        # item_1_T0 = self.df.loc[self.current_step - 0, "High"].item()
        # item_2_T0 = self.df.loc[self.current_step - 0, "Low"].item()
        # item_3_T0 = self.df.loc[self.current_step - 0, "Close"].item()
        # item_4_T0 = self.df.loc[self.current_step - 0, "Volume"].item()
        # item_5_T0 = self.df.loc[self.current_step - 0, "VWAP"].item()

        col_list = list(self.df.columns)
        item_list = []
        for i in range(len(col_list)):
            exec(f'item_{i}_T0 = self.df.loc[{self.current_step} - 0, "{col_list[i]}"].item()')
            exec(f'item_list.append(item_{i}_T0)')
        env_4 = 1 if self.long_short_ratio else 0
        item_list.append(env_4)
        
        obs = np.array(item_list)
        
        return obs

    # Set the current price to a random price within the time step
    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, "Close_Price"].item()
        next_price = self.df.loc[self.current_step + 1, "Close_Price"].item()
        next_return = next_price / current_price - 1
        
        # Go Long
        if action == 0:
            self.net_profit += self.available_balance * PERCENT_CAPITAL * next_return
            self.available_balance += self.net_profit
            self.num_trades_long += 1
                
        # Go Short
        if action == 1:
            self.net_profit += self.available_balance * PERCENT_CAPITAL * -next_return
            self.available_balance += self.net_profit
            self.num_trades_short += 1
        
        # Update metrics
        self.long_short_ratio = self.num_trades_long / (self.num_trades_long + self.num_trades_short)
        self.volatility = self.df.loc[self.current_step - self.lag, "Close_Price"].sum()

    # Execute one time step within the environment
    def step(self, action):
        self._take_action(action)

        reward = self._calculate_reward()
    
        self.current_step += 1
        
        is_max_steps_taken = self.current_step >= self.max_steps - self.lag - 1
        done = True if is_max_steps_taken else False
        
        obs = self._next_observation()

        return obs, reward, done, {}

    # Reset the state of the environment to an initial state
    def reset(self):
        self.available_balance = self.initial_balance
        self.net_profit = 0
        self.current_step = self.lag
        self.num_trades_long = 0
        self.num_trades_short = 0
        self.num_trades_ratio = 0

        return self._next_observation()

    # Render the environment to the screen
    def render(self, mode='human', close=False):
        pass
# ENVIORNMENT END