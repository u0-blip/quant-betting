#%%
import bz2
import json
import pandas as pd
from datetime import datetime
import numpy as np
import calendar
import os
import pickle
import matplotlib.pyplot as plt




base_dir = './BASIC/2020'
months = ['Jan']
# , 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug'
day_dirs = []

for i, m in enumerate(months):
    _, num_days = calendar.monthrange(2020, i+1)
    day_dirs += ['/'.join([base_dir, m, str(day)]) for day in range(1, num_days+1)[:1]]

f_names = []
for d in day_dirs:
    if os.path.exists(d):
        for f in os.scandir(d):
            f_name = f.path.split('\\')[-1]
            f_names.append(f.path + '\\' + f_name + '.bz2')
    else:
        pass


#%% Getting the dataframe
from time import time


def process_msg_by_line(msg, df_runner_market):
    time = float(msg['pt'])/1000
    mc = msg['mc']

    for mc1 in mc:
        marketId = mc1['id']

        if 'marketDefinition' in mc1:
            event_id = mc1['marketDefinition']['eventId']

            runners = mc1['marketDefinition']['runners']
            for r in runners:
                runnerId = r['id']
                # recording the winner status
                if str(runnerId) + str(marketId) not in df_runner_market.index:
                    df_runner_market.loc[str(runnerId) + str(marketId)] = [marketId, runnerId, None, None]
                
                if r['status'] == 'WINNER':
                    df_runner_market.loc[runnerId, marketId]['win'] = True
                elif r['status'] == 'LOSER':
                    df_runner_market.loc[str(runnerId) + str(marketId)]['win'] = False
        elif 'rc' in mc1: 
            for runner in mc1['rc']:
                runnerId = runner['id']
                bid = float(runner['ltp'])
                if type(df_runner_market.loc[str(runnerId) + str(marketId)]['bid']) != type(None):
                    df_runner_market.loc[str(runnerId) + str(marketId)]['bid'] = df_runner_market.loc[str(runnerId) + str(marketId)]['bid'].append(pd.Series([bid], index=[datetime.fromtimestamp(time)]))
                else:
                    df_runner_market.loc[str(runnerId) + str(marketId)]['bid'] = pd.Series([bid], index=[datetime.fromtimestamp(time)])



def process_msg_by_line1(msg, df_runner_market):
    time = float(msg['pt'])/1000
    mc = msg['mc']

    for mc1 in mc:
        marketId = mc1['id']

        if 'marketDefinition' in mc1:
            event_id = mc1['marketDefinition']['eventId']

            runners = mc1['marketDefinition']['runners']
            for r in runners:
                runnerId = r['id']
                # recording the winner status
                if (runnerId, marketId) not in df_runner_market:
                    df_runner_market[(runnerId, marketId)] = {}
                
                if r['status'] == 'WINNER':
                    df_runner_market[(runnerId, marketId)]['win'] = True
                elif r['status'] == 'LOSER':
                    df_runner_market[(runnerId, marketId)]['win'] = False
        elif 'rc' in mc1: 
            for runner in mc1['rc']:
                runnerId = runner['id']
                bid = float(runner['ltp'])
                if 'bid' in df_runner_market[(runnerId, marketId)]:
                    df_runner_market[(runnerId, marketId)]['bid'] = df_runner_market[(runnerId, marketId)]['bid'].append(pd.Series([bid], index=[datetime.fromtimestamp(time)]))
                else:
                    df_runner_market[(runnerId, marketId)]['bid'] = pd.Series([bid], index=[datetime.fromtimestamp(time)])



re_run = True

df_runner_market = {}

start = time()

for i, f_name in enumerate(f_names):
    if re_run:
        with bz2.open(f_name, 'rb') as bet:
            read = bet.read().decode('utf-8')
            read_lines = [ele.strip() for ele in read.split('\n')]
            for line in read_lines:
                if len(line) > 0:
                    msg = json.loads(line)
                    process_msg_by_line1(msg, df_runner_market)

runner_market = pd.DataFrame.from_dict(df_runner_market, orient='index')
runner_market.index.names = ('runnerId', 'marketId')

end = time()
print('total time spend ', end - start)

#%%



#%% get the ones with the most favourable odds
market_list = runner_market.index.unique(level='marketId')
# runner_list = df_runner_market.runnerId.unique()

winnings = []
for m in market_list:
    runner_market.xs(m, level='marketId')


#%% Processing the dataframe

if re_run:
    with open('df_runner_market.pickle', 'wb') as f:
        pickle.dump(df_runner_market, f)
else:
    with open('df_runner_market.pickle', 'rb') as f:
        [df_runner_market, _] = pickle.load(f)

#%% calculating winnings

winnings = []
def cal_winnings(df_runner_market):
    winning = -1
    for id_r, runner in runner_to_market.items():
        for id_m, market in runner.items():
            if 'status' in market and 'fav_odds' in market_to_runner[id_m] and market_to_runner[id_m]['fav_odds'] == id_r:
                if market['status'] and 'price' in market:
                    winning += market['price'][-1] 
                else:
                    winning -= 0
    return winning


winning = cal_winnings(df_runner_market)
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

# %%
