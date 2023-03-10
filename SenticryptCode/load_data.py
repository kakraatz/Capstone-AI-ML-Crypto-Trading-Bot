# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:46:21 2023

@author: kakraatz
"""

import json
import pandas as pd

def load_data():
    path = '../SenticryptCode/sentiment_df.json'
    file = open(path)
    data = json.load(file)
    sentiment_df = pd.json_normalize(data)
    file.close()
    return sentiment_df