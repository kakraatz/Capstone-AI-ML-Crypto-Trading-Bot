# https://senticrypt.com/docs.html

import requests
import json
from datetime import datetime


def get_sentiment():
    date = datetime.today().strftime('%Y-%m-%d_%H')
    req = requests.get('http://api.senticrypt.com/v1/history/bitcoin-' + date + '.json')
    print(req.json())


print(get_sentiment())
