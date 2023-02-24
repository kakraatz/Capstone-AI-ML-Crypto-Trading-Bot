import json
import requests
import time
import datetime as dt

start_date = dt.datetime(2020, 6, 1, 23)
end_date = dt.datetime(2023, 2, 10, 23)
hour_delta = dt.timedelta(days=1)
while start_date <= end_date:
    format_date = str(start_date.strftime('%Y-%m-%d'))
    hour_time = str(start_date.strftime('%Y-%m-%d_%H'))
    url = f'http://api.senticrypt.com/v1/history/bitcoin-{hour_time}.json'
    req = requests.get(url)
    if req.status_code == 200:
        data = json.loads(req.text)
    else:
        data = {"No data for this date"}
    print(f"Appending {hour_time} data to result.")
    time.sleep(1)
    print(f'Creating file for {format_date}')
    with open(format_date + '.json', 'w+') as outfile:
        outfile.write(json.dumps(data, indent=2, default=str))
    start_date += hour_delta
print('Done.')
