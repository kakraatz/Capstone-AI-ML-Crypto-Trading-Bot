import json
import requests
import time
import random
import datetime as dt
import numpy as np

start_date = dt.datetime(2020, 6, 1, 0)
end_date = dt.datetime(2023, 2, 10, 23)
hour_delta = dt.timedelta(hours=1)
result = []
while start_date <= end_date:
    format_date = str(start_date.strftime('%Y-%m-%d'))
    hour_time = str(start_date.strftime('%Y-%m-%d_%H'))
    month = str(start_date.strftime('%Y-%m'))
    next_month = str(np.datetime64(month) + np.timedelta64(1, 'M'))
    url = f'http://api.senticrypt.com/v1/history/bitcoin-{hour_time}.json'
    req = requests.get(url)
    if req.status_code == 200:
        data = json.loads(req.text)
    else:
        data = {"No data for this date"}
    insert = {hour_time: data}
    print(f"Appending {hour_time} data to result.")
    result.append(insert)
    time.sleep(1)
    start_date += hour_delta
    check_month = str(start_date.strftime('%Y-%m'))
    # print(f"month is {month}, check month is {check_month}, next month is {next_month}")
    if check_month == next_month:
        print(f'Creating file for {month}')
        with open(month + '.json', 'w+') as outfile:
            outfile.write(json.dumps(result, indent=2, default=str))
        result.clear()
with open('last.json', 'w+') as outfile:
    outfile.write(json.dumps(result, indent=2, default=str))
print('Done.')
