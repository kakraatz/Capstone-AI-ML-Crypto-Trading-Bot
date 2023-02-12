# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 12:44:46 2023

@author: JohnMurphy
"""
import json
import requests
import pandas as pd
import datetime

class BinanceAPI():
    def __init__(self, token=None):
        self.api_root = 'https://api.binance.us/api/v3/'
        self.token = token
        
    def get_daily_price_history(self, symbol, start_date, end_date, dtype='json'):
        start_dt_obj = datetime.datetime.strptime(start_date,
                               '%Y-%m-%d')        
        start_ms = int(start_dt_obj.timestamp() * 1000)
        end_dt_obj = datetime.datetime.strptime(end_date,
                               '%Y-%m-%d')     
        end_ms = int(end_dt_obj.timestamp() * 1000)
        request_url = self.api_root + 'klines?symbol=' + symbol + '&interval=1d&startTime=' + str(start_ms) + '&endTime=' + str(end_ms)
        api_response = requests.get(request_url)
        if api_response.status_code == 200:            
            json_obj = json.loads(api_response.text)
            if dtype == 'dataframe':
                df = self.__json_to_dataframe(json_obj)
                return df
            else:                
                return json_obj
            
    def __json_to_dataframe(self, data):
        record_array = []
        for index in range(len(data)):
            stick = CandleStick(data[index])
            record_array.append(stick.to_record())
        df = pd.DataFrame(record_array)
        return df
    
    def get_server_time(self):
        request_url = self.api_root + 'time'
        api_response = requests.get(request_url)
        if api_response.status_code == 200:
            json_obj = json.loads(api_response.text)
            return json_obj
            
            
            
class CandleStick():
    def __init__(self, data):
        self.open_time = datetime.datetime.fromtimestamp(data[0]/1000) #pd.to_datetime(data[0], 'ms')
        self.open = float(data[1])
        self.high = float(data[2])
        self.low = float(data[3])
        self.close = float(data[4])
        self.volume = float(data[5])
        self.close_time = datetime.datetime.fromtimestamp(data[6]/1000)
        self.quote_asset_volume = float(data[7])
        self.trade_quantity = data[8]
        self.taker_buy_base_volume = data[9]
        self.taker_buy_quote_volume = data[10]
        self.ignore_flag = data[11]
        
    def to_array(self):
        return [self.open_time, self.open, self.high, self.low, self.close, \
                self.volume, self.close_time, self.quote_asset_volume, \
                    self.trade_quantity, self.taker_buy_base_volume, self.taker_buy_quote_volume, \
                        self.ignore_flag]
    
    def headers(self):
        return ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', \
                'Quote Asset Volume', 'Trade Quantity', 'Taker Buy Base Volume', 'Taker Buy Quote Volume'\
                'Ignore']
    
    def to_record(self):
        record = {
                'Open_Time':self.open_time,
                'Open':self.open,
                'High':self.high,
                'Low':self.low,
                'Close':self.close,
                'Volume': self.volume,
                'Close_Time': self.close_time,
                'Quote_Asset_Volume': self.quote_asset_volume,
                'Trade_Quantity': self.trade_quantity,
                'Taker_Buy_Base_Volume': self.taker_buy_base_volume,
                'Taker_Buy_Quote_Volume': self.taker_buy_quote_volume,
                'Ignore': self.ignore_flag
            }
        return record
    
    