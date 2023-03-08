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
from krm_lib.services.machinelearning.rl.agents.ppo_agent import ActorNetwork
import time

# ENVIORNMENT START
# Initialise variables
ALPHA = 0.0004
MAX_INT = 2147483647
MAX_TRADES = 10000
MAX_OPEN_POSITIONS = 1
PERCENT_CAPITAL = 0.25
KILL_THRESH = 0.4 # Threshold for balance preservation

# Structure environment
class TestTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_account_balance=1000, trading_cost_rate=0.001):
        super(TestTradingEnv, self).__init__()
        
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
        
        item_0_T0 = self.df.loc[self.current_step - 0, "Open"].item()
        item_1_T0 = self.df.loc[self.current_step - 0, "High"].item()       
        item_2_T0 = self.df.loc[self.current_step - 0, "Low"].item()
        item_3_T0 = self.df.loc[self.current_step - 0, "Close"].item()
        item_4_T0 = self.df.loc[self.current_step - 0, "Volume"].item()
        item_5_T0 = self.df.loc[self.current_step - 0, "VWAP"].item()
        
        env_4 = 1 if self.long_short_ratio else 0
        
        obs = np.array([item_0_T0, item_1_T0, item_2_T0, item_3_T0, item_4_T0, item_5_T0, env_4])
        
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

####### START ENVIORNMENT #########
class RealTradingEnv(gym.Env):
####### END ENVIORNMENT #########
    """ Coin Trading enviornment w/ OPEN AI gym """
    metadata = {'rendor.modes':['human']}
    
    def __init__(self, df, initial_account_balance=1000, trading_cost_rate=0.004, window=5):
        super(RealTradingEnv, self).__init__()
        
        # Observation Space -> Historic window
        self.window = window
        self.window_parameters = 6
        self.space_shape = window * self.space_parameters + 2
        
        # Pass Generic Variable as a Pandas Dataframe
        self.df = df
        
        # Account Variables
        self.initial_balance = initial_account_balance
        self.available_balance = initial_account_balance
        self.net_worth = initial_account_balance
        
        
        self.trading_cost_rate = trading_cost_rate
        self.realized_profit = 0
        self.unrealized_profit = 0
        self.last_profit = 0
        
        # Position variables
        self.open_quantities = []
        self.open_prices = []
        self.trading_costs = 0
        self.open_positions = 0
        self.closed_positions = 0
        self.incorrect_position_calls = 0
        self.num_trades = 0
        self.held_for_period = 0
        
        # Current Step
        self.current_step = 0
        self.first_decision_step = 5
        self.max_steps = len(df)

        # Actions of the format Long, Hold, Close
        # Actions/Decisions are Discrete - Long, Hold, Close 
        self.action_space = spaces.Discrete(3)

        # Prices contains the Close and Close Returns etc
        # Observations (input data) is continuous
        #self.observation_space = spaces.Box(low=-1, high=1, shape=(8, ), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.space_shape, ), dtype=np.float32)
        
    # REWARD STRUCTURE
    # NOTE: CAN and Should try to come up with a better reward structure if possible...
    def _calculate_reward(self):
        reward = 0
        if self.num_trades:
            # Incentivize for profit: rewarding based on "profit-per-trade" - encourage not to over-trade
            reward += self.realized_profit / self.num_trades
            # Incentivize less for 'unrealized' profit (0.30 percent)
            reward += self.unrealized_profit / self.num_trades * 0.3
            # Add 1 to reward if it made a profit on most recent trade
            reward += 1 if self.last_profit > 0 else 0
            # Remove 1 from reward if incorrect position calls is > 0
        reward -= 1 if self.incorrect_position_calls > 0 else 0
        # Penialize if AI does not 'trade'
        if reward <= 0:
            reward -= 2
        return reward
    
    # Structure sign observation data
    # Revisit - Observation structure to add additional technical indicators for Crypto Trading
    def _next_observation(self):
        
        current_position = 1 if self.open_positions else 0
        num_trades = self.num_trades / len(self.df) if self.num_trades > 0 else 0
        
        ''' Sin Wave Observation
        obs = np.array([close_item, close_rt_item, close_T1_item, close_T2_item, close_T3_item, close_T4_item, 
                        current_position, num_trades])
        '''
        
        ''' Stock observation VWAP -> No memory of the past '''
        # Looks at 6 days of history data
        obs_array = []
        for x in range(0, self.window):
            item_open = self.df.loc[self.current_step - x, "Open"].item()
            item_high = self.df.loc[self.current_step - x, "High"].item()       
            item_low = self.df.loc[self.current_step - x, "Low"].item()
            item_close = self.df.loc[self.current_step - x, "Close"].item()
            item_volume = self.df.loc[self.current_step - x, "Volume"].item()
            item_VWAP = self.df.loc[self.current_step - x, "VWAP"].item()
            obs_array.append(item_open)
            obs_array.append(item_high)
            obs_array.append(item_low)
            obs_array.append(item_close)
            obs_array.append(item_volume)
            obs_array.append(item_VWAP)
        obs_array.append(current_position)
        obs_array.append(num_trades)
        obs = np.array(obs_array)        
        
        return obs
    
    # Calculate current open value
    def _calculate_open_value(self):
        open_trades_value = 0
        counts = 0
        for qty in self.open_quantities:
            acquisition_price = self.open_prices[counts]
            open_trades_value += acquisition_price * qty
            counts += 1
        return open_trades_value
        
    # Calculate gross profit
    def _profit_calculation(self, current_price, calc_type):
        open_trades_value = self._calculate_open_value()
        total_quantity_held = sum(self.open_quantities)
        current_value = total_quantity_held * current_price
        gross_profit = current_value - open_trades_value
        
        if calc_type == "close_position":
            trading_costs = current_value * self.trading_cost_rate
            self.trading_costs += trading_costs
        elif calc_type == "open_position" or calc_type == "hold_position":
            trading_costs = open_trades_value * self.trading_cost_rate
        
        net_profit = gross_profit - trading_costs
        
        return net_profit

    # ACTION FUNCTIONS 
    # Set the current price to a random price within the time step
    def _take_action(self, action):
        #current_price = self.df.loc[self.current_step, "Close"].item()
        current_price = self.df.loc[self.current_step, "Close_Price"].item()
        # Reset last profit
        self.last_profit = 0
        self.incorrect_position_calls = 0
        
        action_string = "None"
        current_netprofit = 0
        # Go Long
        if action == 0:
            if self.open_positions < MAX_OPEN_POSITIONS:
                action_string = "Long_Open"
                net_profit = self._profit_calculation(current_price, "open_position")
                net_worth = self.net_worth + net_profit
                trading_allowance = net_worth * PERCENT_CAPITAL
                
                self.open_quantities.append(trading_allowance / current_price)
                self.open_prices.append(current_price)
                self.trading_costs += trading_allowance * self.trading_cost_rate
                self.num_trades += 1
            else:
                self.incorrect_position_calls += 1
                action_string = "Long_MAXPos"

        # Hold Positions
        if action == 1: 
            action_string = "Hold"
            net_profit = self._profit_calculation(current_price, "hold_position")
            self.unrealized_profit += net_profit
            if self.open_positions > 0:
                self.held_for_period += 1
                
        # Close Positions
        if action == 2:
            if self.open_positions != 0:                
                action_string = "Close"
                net_profit = self._profit_calculation(current_price, "close_position")
                self.last_profit = net_profit
                self.realized_profit += net_profit
                self.unrealized_profit = 0
                self.open_quantities = []
                self.open_prices = []
                self.held_for_period = 0
                self.closed_positions += 1
            else:
                action_string = "Close_NoOpen"
                self.incorrect_position_calls += 1
        
        # Update variables
        open_trades_value = self._calculate_open_value()
        self.open_positions = len(self.open_quantities)
        self.net_worth = self.initial_balance + self.unrealized_profit + self.realized_profit
        self.available_balance = self.initial_balance - open_trades_value + self.realized_profit        
        total_quantity_held = sum(self.open_quantities)
        current_value = total_quantity_held * current_price
        #print(f"step: {self.current_step}, action: {action_string}, current_price: {current_price}, current_worth: {self.net_worth}, realized_p: {self.realized_profit}, open_positions_held: {self.open_positions}, quantity_held:{total_quantity_held}, open_original_value:{open_trades_value}, open_current_value:{current_value}")
        print(f"{self.current_step}|{action_string}|{current_price}|{self.net_worth}|{self.realized_profit}|{self.open_positions}|{total_quantity_held}|{open_trades_value}|{current_value}")

    # Execute one time step within the environment
    def step(self, action):
        #previous_net_worth = self.net_worth
        self._take_action(action)
        
        reward = self._calculate_reward()
    
        self.current_step += 1
        
        is_max_steps_taken = self.current_step >= self.max_steps - 1
        is_account_balance_reached = self.net_worth <= self.initial_balance * KILL_THRESH
        #print(f"is_account_balance_reached: {is_account_balance_reached}, net_worth: {self.net_worth}, kill: {self.initial_balance * KILL_THRESH}")
        done = True if is_max_steps_taken or is_account_balance_reached else False
        #print(f"is_account_balance_reached: {is_account_balance_reached}, net_worth: {self.net_worth}, kill: {self.initial_balance * KILL_THRESH}")
        obs = self._next_observation()
        #print(f"Out of step -> obs: {obs}, reward: {reward}, done: {done}")
        return obs, reward, done, {}

    # Reset the state of the environment to an initial state
    def reset(self, reset_index=0):
        self.account_balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.realized_profit = 0
        self.unrealized_profit = 0
        self.open_quantities = []
        self.open_prices = []
        self.trading_costs = 0
        self.open_positions = 0
        self.incorrect_position_calls = 0
        self.current_step = reset_index + 10

        return self._next_observation()

    # Render the environment to the screen
    def render(self, mode='human', close=False):
        profit = self.net_worth - self.initial_balance
        return profit
    
    def run_simulation(self, saved_model_path="tmp/actor_torch_ppo", df_start_index=0):
        n_actions = self.action_space.n
        input_dims = self.observation_space.shape
        model = ActorNetwork(n_actions, input_dims, ALPHA)
        model.load_state_dict(torch.load(saved_model_path))
        model.eval()
        
        # Run Simulation
        reporting_df = self.df.copy()
        
        n_steps = 0
        obs = self.reset(reset_index=df_start_index) 
        state = torch.tensor(obs).float().to(model.device)
        dist = model(state)
        probs = dist.probs.cpu().detach().numpy()  
        
        done = False
        score = 0
        action = np.argmax(probs)
        while not done:
            action_name = "None"
            if action == 0:
                action_name = "Long"
            elif action == 1:
                action_name = "Hold"
            else:
                action_name = "Close"            
            current_price = self.df.loc[self.current_step, "Close_Price"].item()
            index = self.df.loc[self.current_step, "index"].item()
            #print(action_name, np.argmax(probs), index, current_price, self.net_worth)
            time.sleep(0.5)
            state = torch.tensor(obs).float().to(model.device)
            dist = model(state)
            probs = dist.probs.cpu().detach().numpy()
            action = np.argmax(probs)
            #action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = self.step(action)
            n_steps += 1
            score += reward
            #agent.remember(observation, action, prob, val, reward, done)
            obs = observation_
        print("Done at step: " + str(n_steps))
        return self.df
        
