import bz2
import json
import pandas as pd
from datetime import datetime
import numpy as np
import calendar
import os
import pickle
import matplotlib.pyplot as plt


def process_msg(msg):
    time = float(msg['pt'])/1000
    mc = msg['mc']

    for mc1 in mc:
        market_id = mc1['id']

        initial_add_runner = True
        if market_id in market_to_runner:
            initial_add_runner = False
        else:
            market_to_runner[market_id] = {
                'runners' : set()
            }

        if 'marketDefinition' in mc1:
            event_id = mc1['marketDefinition']['eventId']

            runners = mc1['marketDefinition']['runners']
            for r in runners:
                # recording the runners
                if r['id'] not in runner_to_market:
                    runner_to_market[r['id']] = {}
                    runner_to_market[r['id']][market_id] = {}
                elif market_id not in runner_to_market[r['id']]:
                    runner_to_market[r['id']][market_id] = {}

                # recording the winner status
                if r['status'] == 'WINNER':
                    runner_to_market[r['id']][market_id]['status'] = True
                elif r['status'] == 'LOSER':
                    runner_to_market[r['id']][market_id]['status'] = False
                
                # recording the markets
                if initial_add_runner:
                    market_to_runner[market_id]['runners'].add(r['id'])

        elif 'rc' in mc1: 
            for runner in mc1['rc']:
                runner_id = runner['id']
                price = float(runner['ltp'])
                if runner_id in runner_to_market and market_id in runner_to_market[runner_id]:
                    if len(runner_to_market[runner_id][market_id]) == 0:
                        runner_to_market[runner_id][market_id]['price'] = pd.Series([price], index=[datetime.fromtimestamp(time)])
                    else:
                        runner_to_market[runner_id][market_id]['price'] = runner_to_market[runner_id][market_id]['price'].append(pd.Series([price], index=[datetime.fromtimestamp(time)]))
                else:
                    print('runner ', runner_id, ' doesn\'t appear in the market def')

base_dir = './BASIC/2020'
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
day_dirs = []
dates = []
for i, m in enumerate(months):
    _, num_days = calendar.monthrange(2020, i+1)
    day_dirs += ['/'.join([base_dir, m, str(day)]) for day in range(1, num_days+1)]
    dates += pd.to_datetime(['/'.join([str(i+1), str(day), '2020']) for day in range(1, num_days+1)])

winnings = []

f_names = []
for d in day_dirs:
    if os.path.exists(d):
        for f in os.scandir(d):
            f_name = f.path.split('\\')[-1]
            f_names.append(f.path + '\\' + f_name + '.bz2')
    else:
        pass

# f_names = ['./BASIC/2020/Jan/1/29636351/29636351.bz2']


re_run = False

for i, f_name in enumerate(f_names):
    res_fname = '.'.join(f_name.split('.')[:-1]) + '.res'
    if re_run or not os.path.exists(res_fname):
        runner_to_market = {}
        market_to_runner = {}

        with bz2.open(f_name, 'rb') as bet:
            read = bet.read().decode('utf-8')
            read_lines = [ele.strip() for ele in read.split('\n')]
            # print()
            try:
                for line in read_lines:
                    if len(line) > 0:
                        msg = json.loads(line)
                        process_msg(msg)
            except Exception:
                print('there is an exception', line)

        for market_id, race in market_to_runner.items():
            last_odds = []
            runners = race['runners']
            market_to_runner[market_id]['runners'] = list(runners)
            for runner in runners:
                if runner in runner_to_market and market_id in runner_to_market[runner] and 'price' in runner_to_market[runner][market_id]:
                    last_odds.append({
                        runner_to_market[runner][market_id]['price'][-1]
                        })
                else:
                    # print('market runner comb doens\'t exist', runner, market_id)
                    pass
            if len(last_odds) > 0:
                fav_odds = np.argmin(last_odds)
                market_to_runner[market_id]['fav_odds'] = market_to_runner[market_id]['runners'][fav_odds]

        winning = 0
        for id_r, runner in runner_to_market.items():
            for id_m, market in runner.items():
                if 'status' in market and 'fav_odds' in market_to_runner[id_m] and market_to_runner[id_m]['fav_odds'] == id_r:
                    if market['status'] and 'price' in market:
                        winning += market['price'][-1] - 1
                    else:
                        winning -= 1
                    # print('winning is: ', winning)

        with open(res_fname, 'wb') as f:
            pickle.dump([runner_to_market, market_to_runner, winning], f)
    else:
        winning = 0
        # print('Already ran, printing results: ')
        with open(res_fname, 'rb') as f:
            [runner_to_market, market_to_runner, _] = pickle.load(f)

        for id_r, runner in runner_to_market.items():
            for id_m, market in runner.items():
                if 'status' in market and 'fav_odds' in market_to_runner[id_m] and market_to_runner[id_m]['fav_odds'] == id_r:
                    if market['status'] and 'price' in market:
                        winning += market['price'][-1] - 1
                    else:
                        winning -= 1


    winnings.append(winning)
        # print('winning is: ', winning)

with open('./all.res', 'w') as f:
    json.dump(winnings, f)

print('total: ', np.sum(winnings))
plt.plot(winnings)
plt.show()

# print(market_to_runner)
# print(runner_to_market)

# print(winning)
