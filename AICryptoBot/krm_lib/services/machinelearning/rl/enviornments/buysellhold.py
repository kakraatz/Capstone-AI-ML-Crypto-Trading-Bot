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
from krm_lib.utilities.helpers import Plotters
import time

# ENVIORNMENT START
# Initialise variables
ALPHA = 0.0004
MAX_INT = 2147483647
MAX_TRADES = 10000
MAX_OPEN_POSITIONS = 1
PERCENT_CAPITAL = 0.25
KILL_THRESH = 0.4 # Threshold for balance preservation


####### START ENVIORNMENT #########
class BuySellHoldTradingEnv(gym.Env):
####### END ENVIORNMENT #########
    """ Coin Trading enviornment w/ OPEN AI gym """
    metadata = {'rendor.modes':['human']}
    
    def __init__(self, df, initial_account_balance=1000, trading_cost_rate_maker=0.0035, trading_cost_rate_taker=0.0045, window=5):
        super(BuySellHoldTradingEnv, self).__init__()
        self.simulation = False
        self.first_decision_step = window
        # Observation Space -> Historic window
        self.window = window
        self.space_parameters = 6
        self.space_shape = window * self.space_parameters + 1 # add 1 to include current_open_opportunity
        
        # Pass Generic Variable as a Pandas Dataframe
        self.df = df
        
        # Account Variables
        self.initial_balance = initial_account_balance
        self.available_balance = initial_account_balance
        self.net_worth = initial_account_balance
        
        
        #self.trading_cost_rate = trading_cost_rate
        self.trading_cost_rate_maker = trading_cost_rate_maker
        self.trading_cost_rate_taker =  trading_cost_rate_taker
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
        self.max_steps = len(df)
        
        # Actions
        self.current_action = 1  #default action is hold
        self.previous_action = 1 #default action is hold
        
        # previous, current, next price
        self.previous_price  = 0
        self.next_price = 0
        self.current_price = 0
        
        # opportunity will be used to calculate "reward"
        self.previous_opportunity = 0
        self.current_opportunity = 0
        self.next_opportunity = 0 # might not want to use this in reward
        self.total_positive_opportunity = 0
        
        # Actions of the format Long, Hold, Close
        # Actions/Decisions are Discrete - Long, Hold, Close 
        self.action_space = spaces.Discrete(3)

        # Prices contains the Close and Close Returns etc
        # Observations (input data) is continuous
        #self.observation_space = spaces.Box(low=-1, high=1, shape=(8, ), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.space_shape, ), dtype=np.float32)
        
        # Simulation Arrays
        self.initial_investment = initial_account_balance * PERCENT_CAPITAL
        self.initial_quantity_market = 0
        self.bot_profit = []
        self.market_profit = []
        
        
    # REWARD STRUCTURE
    # NOTE: CAN and Should try to come up with a better reward structure if possible...

    '''
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
    '''
    
    def _calculate_reward(self):
        reward = 0
        if True:
            # OPEN - Reward Structure
            if self.current_action == 0:
                # 1. Next Opportunity Reward -> Reward based on next_opportunity -> standard reward
                if self.next_opportunity > 0:      
                    reward += self.next_opportunity * 1.0            # (POSITIVE REWARD) add next opportunity to reward -> next opportunity is positive the 
                else:
                    reward += self.next_opportunity * 1.0            # (NEGATIVE REWARD) - bad "open" call: make negitive reward more significant
                
                # 2. Current Opportunity Reward -> Current opportunity was a 'missed' opportunity
                if self.current_opportunity > 0:
                    reward += -self.current_opportunity * 0.50        # (NEGATIVE REWARD) - current opportunity is not a 'realized' opportunity (it was actually missed). 
                    
                else:                                                 # (POSITIVE Reward)
                    reward += -self.current_opportunity * 0.50 
                
                # 3. Market Reversal Reward -> Found "Valley"     
                if self.next_opportunity > 0:
                # Encourage - calling "Open" if reversal is a 'valley'
                    if self.current_opportunity < 0:
                        reward += -self.current_opportunity
                        if self.previous_opportunity < 0:
                            reward += -self.previous_opportunity
                # Discourage - calling "Open" if reversal is a 'peak'
                                            
                # 4. Trading Cost Reward -> negitive impact on reward
                reward += -self.trading_cost_rate_maker
                
            # HOLD - Reward Structure
            elif self.current_action == 1:
                # Current Opportunity Reward -> show be primary reward for HOLD Call
                if self.next_opportunity > 0:
                    reward += self.next_opportunity * 1.15     # (POSITIVE REWARD) - scaled positive reward incentized to hold if the sentiment is positive
                else:
                    reward += self.next_opportunity * 5         # (NEGATIVE REWARD) - Negative incentized to hold if the sentiment is positive
                    
            # CLOSE - Reward Structure
            elif self.current_action == 2:
                # 1. Next Opportunity Reward -> Reward based on next_opportunity -> standard reward
                if self.next_opportunity > 0:      
                    reward += -self.next_opportunity  * 1.0      # (NEGATIVE REWARD) add next opportunity to reward -> next opportunity is positive the 
                else:
                    reward += -self.next_opportunity * 1.0          # (POSITIVE REWARD) - next_opportunity is < 0 (negative) -> and you closed before it went negative so add a positive reward by adding (-) next opportunity.
                
                # 2. Current Opportunity Reward -> On ACTION "CLOSE" the current opportuntity is the highest weighted opportunity
                if self.current_opportunity > 0:
                    reward += self.current_opportunity               # (POSITIVE REWARD) - closed when 'current' opporuntity was positive -> this was a 'realized' profit
                    
                else:                                                # (POSITIVE Reward) 
                    reward += self.current_opportunity * 0.25 
                    
                # 3. Market Reversal Reward -> Found "Peak"  
                if self.next_opportunity < 0:
                    # Encourage - calling "Open" if reversal is a 'peak'
                    if self.current_opportunity > 0:
                        reward += self.current_opportunity
                        if self.previous_opportunity > 0:
                            reward += self.previous_opportunity
                            # Possibly structure the logic as below (to really incentize the good behavior)
                            '''
                            if previous_action == 1:
                                reward += self.previous_opportunity
                            else:
                                reward += self.previous_opportunity * 0.25
                            '''
                            
                # 4. Trading Cost Reward -> negitive impact on reward
                reward += -self.trading_cost_rate_taker       
        return reward * 100
    
    def _calculate_reward_improved(self):
        reward = 0
        if True:
            # OPEN - Reward Structure
            if self.current_action == 0:
                if self.open_positions < MAX_OPEN_POSITIONS:
                    # 1. Next Opportunity Reward -> Reward based on next_opportunity -> standard reward
                    if self.next_opportunity > 0:      
                        reward += self.next_opportunity * 1.0            # (POSITIVE REWARD) add next opportunity to reward -> next opportunity is positive the 
                    else:
                        reward += self.next_opportunity * 1.0            # (NEGATIVE REWARD) - bad "open" call: make negitive reward more significant
                else:
                    # Always discourage calling "open" if the next_opportunity is negative
                    if self.next_opportunity < 0:
                        reward += self.next_opportunity # (Negitive Reward)
                    else:
                        reward += self.next_opportunity * 0.2 # (Small Positive Reward)
                # 4. Trading Cost Reward -> negitive impact on reward
                reward += -self.trading_cost_rate_maker + -self.trading_cost_rate_taker
                
            # HOLD - Reward Structure
            elif self.current_action == 1:
                # Reward only if there actually is an open_position
                if self.open_positions: 
                # Current Opportunity Reward -> show be primary reward for HOLD Call
                    if self.next_opportunity > 0:
                        reward += self.next_opportunity + self.trading_cost_rate_taker + self.trading_cost_rate_maker # (extra credit because you saved the cost of trading)    # (POSITIVE REWARD) - scaled positive reward incentized to hold if the sentiment is positive
                    else:
                        reward += self.next_opportunity         # (NEGATIVE REWARD) - Negative incentized to hold if the sentiment is positive
                else:
                    # No open positions -> always set a negitive reward on no open positions
                    # don't call  hold when their is no open positions
                    if self.next_opportunity > 0:
                        reward += -self.next_opportunity
                    else:
                        reward += self.next_opportunity
                        
            # CLOSE - Reward Structure
            elif self.current_action == 2:
                # Reward only if there actually is an open_position
                if self.open_positions:
                    reward += self._calculate_open_opportunity()
                    reward += -self.trading_cost_rate_taker
                    reward += -self.trading_cost_rate_maker # If you are closing a position you need to open one later
                else:
                    if self.next_opportunity > 0:
                        reward += -self.next_opportunity * 0.5
                    else:
                        reward += -self.next_opportunity
                        
        return reward * 100    
    
    # Structure sign observation data
    # Revisit - Observation structure to add additional technical indicators for Crypto Trading
    def _next_observation(self):
        
        current_open_opportunity = 0 
        if self.open_positions:
            current_open_opportunity = self._calculate_open_opportunity()
        
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
        obs_array.append(current_open_opportunity) # if quanity is open there is a defined opportunity
        obs = np.array(obs_array)        
        
        return obs
    
    def _calculate_open_opportunity(self):
        open_value = self._calculate_open_value()
        current_value = sum(self.open_quantities) * self.current_price
        return (current_value - open_value) / current_value
        
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
            trading_costs = current_value * self.trading_cost_rate_taker
            self.trading_costs += trading_costs
        elif calc_type == "open_position":
            trading_costs = open_trades_value * self.trading_cost_rate_maker
        elif calc_type == "hold_position":
            trading_costs = 0
        
        net_profit = gross_profit - trading_costs
        
        return net_profit

    # ACTION FUNCTIONS 
    # Set the current price to a random price within the time step
    def _take_action(self, action):
        #previous_net_worth = self.net_worth
        #current_price = self.df.loc[self.current_step, "Close"].item()
        self.current_price = self.df.loc[self.current_step, "Close_Price"].item()
        # Reset last profit
        self.last_profit = 0
        self.incorrect_position_calls = 0        
        action_string = "None"
        #current_netprofit = 0
        # Go Long
        if action == 0:
            if self.open_positions < MAX_OPEN_POSITIONS:
                action_string = "Long_Open"
                net_profit = self._profit_calculation(self.current_price, "open_position")
                net_worth = self.net_worth + net_profit
                trading_allowance = net_worth * PERCENT_CAPITAL
                
                self.open_quantities.append(trading_allowance / self.current_price)
                self.open_prices.append(self.current_price)
                self.trading_costs += trading_allowance * self.trading_cost_rate_maker
                self.num_trades += 1
            else:
                self.incorrect_position_calls += 1
                action_string = "Long_MAXPos"

        # Hold Positions
        if action == 1: 
            action_string = "Hold"
            net_profit = self._profit_calculation(self.current_price, "hold_position")
            #print(f"netprofit: {net_profit}")
            self.unrealized_profit = net_profit
            if self.open_positions > 0:
                self.held_for_period += 1
                
        # Close Positions
        if action == 2:
            if self.open_positions != 0:                
                action_string = "Close"
                net_profit = self._profit_calculation(self.current_price, "close_position")
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
        current_value = total_quantity_held * self.current_price
        if self.simulation:
            #print(f"step: {self.current_step}, action: {action_string}, current_price: {current_price}, current_worth: {self.net_worth}, realized_p: {self.realized_profit}, open_positions_held: {self.open_positions}, quantity_held:{total_quantity_held}, open_original_value:{open_trades_value}, open_current_value:{current_value}")
            #print(f"{self.current_step}|{action_string}|{current_price}|{self.net_worth}|{self.realized_profit}|{self.open_positions}|{total_quantity_held}|{open_trades_value}|{current_value}")
            print(f"{self.current_step}|{action_string}|{self.current_price}|{self.net_worth} = {self.initial_balance} + {self.unrealized_profit} + {self.realized_profit}  |{open_trades_value}|{current_value}")
            #print(f"{self.current_step}|{action_string}|{current_price}|{self.previous_opportunity * 100}|{self.current_opportunity * 100}|{self.next_opportunity * 100}|{self.total_positive_opportunity * 100}|")
    
    def _calculate_opportunity(self, days=0):
        current_index = self.current_step + days
        previous_index = self.current_step + days - 1
        opportunity = 0
        try:
            current_price = self.df.loc[current_index, "Close_Price"].item()
            previous_price = self.df.loc[previous_index, "Close_Price"].item()
            opportunity = (current_price - previous_price) / current_price
        except Exception:
            pass
        return opportunity
        
    
    # Execute one time step within the environment
    def step(self, action): 
        self.current_action = action
        self._take_action(action)
        #reward = self._calculate_reward()   
        reward = self._calculate_reward_improved()
        if self.simulation:
            print(f"-------------> step: {self.current_step}|reward: {reward}")
        self.current_step += 1
        self.previous_opportunity = self._calculate_opportunity(-1)
        self.current_opportunity = self._calculate_opportunity()
        self.next_opportunity = self._calculate_opportunity(1)        
        
        
        if self.current_opportunity > 0:
            self.total_positive_opportunity = self.total_positive_opportunity + self.current_opportunity
        
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
        self.current_step = reset_index + self.window

        return self._next_observation()

    # Render the environment to the screen
    def render(self, mode='human', close=False):
        profit = self.net_worth - self.initial_balance
        return profit
    
    def run_simulation(self, saved_model_path="tmp/actor_torch_ppo_buysellhold", df_start_index=0):
        self.simulation = True
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
        self.initial_price_market = self.df.loc[self.current_step, "Close_Price"].item()
        self.initial_quantity_market = self.initial_investment / self.initial_price_market
        steps_arr = []
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
            
            bot_position = self.initial_investment + self.realized_profit + self.unrealized_profit
            market_position = self.initial_quantity_market * current_price            
            self.bot_profit.append(bot_position)
            self.market_profit.append(market_position)
            steps_arr.append(n_steps)
            state = torch.tensor(obs).float().to(model.device)
            dist = model(state)
            probs = dist.probs.cpu().detach().numpy()
            action = np.argmax(probs)
            self.current_action = action
            #action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = self.step(action)
            n_steps += 1
            score += reward
            obs = observation_
            self.previous_action = action
        print("Done at step: " + str(n_steps))
        plotter = Plotters()
        plotter.plot_benchmark_equity(self.market_profit, self.bot_profit, steps_arr)    
        print(self.market_profit)
        print(self.bot_profit)
        return self.df

