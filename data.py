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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor



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

df_runner_market = pd.DataFrame.from_dict(df_runner_market, orient='index')
df_runner_market.index.names = ('runnerId', 'marketId')
market_list = df_runner_market.index.unique(level='marketId')


end = time()
print('total time spend ', end - start)
#%%
df_runner_market['bid_arr'] = df_runner_market.bid.apply(lambda x: x.to_numpy() if type(x) != float else 0)
df_runner_market['bid_len'] = df_runner_market.bid_arr.apply(len)
fill_bid_arr_size = df_runner_market.describe().loc['75%'].values[0] + 5
df_runner_market['bid_arr_filled'] = df_runner_market.bid_arr.apply(lambda x: np.pad(x, (int(fill_bid_arr_size - len(x)),0), mode='constant', constant_values=0) if fill_bid_arr_size > len(x) else x[int(len(x) - fill_bid_arr_size):])


#%%
# df_runner_market.groupby('marketId').count().bid.sort_values().plot(use_index=False)
# drop everything with bigger than 15 runners
fill_runners = df_runner_market.groupby('marketId').count().bid.sort_values().describe()['75%']
df_runner_market.drop(df_runner_market.groupby('marketId').count().bid[lambda x: x > fill_runners].index, level='marketId')
combined_market = df_runner_market.groupby('marketId')[['bid_arr_filled', 'win']].apply(lambda x:( np.concatenate(x['bid_arr_filled']), x['win'].values))
all_filled = fill_runners*fill_bid_arr_size


#%%

combined_market = pd.DataFrame.from_dict(dict(zip(combined_market.index, combined_market.values)), orient='index', columns=['bid_arr', 'win'])

combined_market.bid_arr = combined_market.bid_arr.apply(lambda x: np.pad(x, (0, int(all_filled - len(x))), mode='constant', constant_values=0) if all_filled >= len(x) else print('Array too long'))

combined_market.win = combined_market.win.apply(lambda x: np.pad(x.astype(float), (0, int(fill_runners - len(x))), mode='constant', constant_values=0) if fill_runners >= len(x) else print('Array too long'))


#%% get the ones with the most favourable odds
train_market = combined_market.iloc[:int(len(combined_market)*0.8)]
test_market = combined_market.iloc[int(len(combined_market)*0.8):]
train_value = np.vstack(train_market.bid_arr.values)
train_label = np.vstack(train_market.win.values)
test_value = np.vstack(test_market.bid_arr.values)
test_label = np.vstack(test_market.win.values)

# runner_list = df_runner_market.runnerId.unique()

#%% use Kneighbors regressor



knn = KNeighborsRegressor()
regr = MultiOutputRegressor(knn)

regr.fit(train_value, train_label)
pred = regr.predict(test_value)

ran = np.random.rand(*test_label.shape)
mse_rand = ((ran - test_label)**2).mean()
mse_pred = ((pred - test_label)**2).mean()
print('rand ', mse_rand, ' pred ', mse_pred)

#%% use xgboost
import xgboost as xgb
# read in data
dtrain = xgb.DMatrix(train_value, train_label)
dtest = xgb.DMatrix(test_value)
# specify parameters via map
param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
num_round = 2


# fitting
xgbr = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0,
       importance_type='gain', learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='reg:squarederror', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1)

multioutputregressor = MultiOutputRegressor(xgbr).fit(train_value, train_label)

# make prediction
xgb_preds = multioutputregressor.predict(test_value)

mse_pred = ((xgb_preds - test_label)**2).mean()
print(' xgb pred mse ', mse_pred)

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
