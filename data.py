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

from sqlitedict import SqliteDict
from time import time

f_name_db = SqliteDict('checkpoints/filenames.sqlite', autocommit=True)

re_run_dir = False
re_run_data = True


def get_month_fname(i, m):
    base_dir = './BASIC/2020'
    _, num_days = calendar.monthrange(2020, i+1)
    f_month_name = []
    for d in ['/'.join([base_dir, m, str(day)]) for day in range(1, num_days+1)]:
        if os.path.exists(d):
            for f in os.scandir(d):
                f_name = f.path.split('\\')[-1]
                f_month_name.append(f.path + '\\' + f_name + '.bz2')

    return f_month_name

print('Start processing files...')
f_name_pickle = 'checkpoints/f_name.pickle'
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']

f_names = []

for i, m in enumerate(months):
    if re_run_dir or m not in f_name_db:
        f_name_db[m] = get_month_fname(i, m)
    f_names += f_name_db[m]

f_name_db.close()

#%% Getting the dataframe
from time import time

def process_msg_by_line1(msg, df_runner_market, df_market):
    time = float(msg['pt'])/1000
    mc = msg['mc']

    for mc1 in mc:
        marketId = mc1['id']

        if 'marketDefinition' in mc1:
            market_def = mc1['marketDefinition']
            event_id = market_def['eventId']
            runners = market_def['runners']

            if marketId not in df_market:
                df_market[marketId] = {
                    'name': market_def['name'] if 'name' in market_def else None,
                    'venue': market_def['venue'] if 'venue' in market_def else None,
                    'eventName': market_def['eventName'] if 'eventName' in market_def else None,
                    'numberOfWinners': market_def['numberOfWinners'] if 'numberOfWinners' in market_def else None
                }
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




runner_market_f = 'checkpoints/runner_market.pickle'

df_runner_market = {}

print('Start processing data...')
start = time()
if re_run_data or not os.path.exists(runner_market_f):
    start = time()

    df_market = {}

    for i, f_name in enumerate(f_names):
        if re_run_data:
            with bz2.open(f_name, 'rb') as bet:
                read = bet.read().decode('utf-8')
                read_lines = [ele.strip() for ele in read.split('\n')]
                for line in read_lines:
                    if len(line) > 0:
                        msg = json.loads(line)
                        process_msg_by_line1(msg, df_runner_market, df_market)

    df_runner_market = pd.DataFrame.from_dict(df_runner_market, orient='index')
    
    df_runner_market.index.names = ('runnerId', 'marketId')
    df_market = pd.DataFrame.from_dict(df_market, orient='index')

    end = time()
    print('total time spend ', end - start)

    df_runner_market.dropna(inplace=True)
    df_runner_market['bid_arr'] = df_runner_market.bid.apply(lambda x: x.to_numpy() if type(x) != float else 0)
    df_runner_market['bid_len'] = df_runner_market.bid_arr.apply(len)
    fill_bid_arr_size = df_runner_market.describe().loc['75%'].values[0] + 5
    df_runner_market['bid_arr_filled'] = df_runner_market.bid_arr.apply(lambda x: np.pad(x, (int(fill_bid_arr_size - len(x)),0), mode='constant', constant_values=0) if fill_bid_arr_size > len(x) else x[int(len(x) - fill_bid_arr_size):])

    with open(runner_market_f, 'wb') as rmf:
        pickle.dump([df_runner_market, df_market], rmf)

else:
    with open(runner_market_f, 'rb') as rmf:
        df_runner_market, df_market = pickle.load(rmf)
        fill_bid_arr_size = df_runner_market.describe().loc['75%'].values[0] + 5

print('data processing took: ', time() - start)


market_list = df_runner_market.index.unique(level='marketId')

#%%
#drop place and focus solely on wins
df_runner_market['runner_id'] = df_runner_market.index.get_level_values(level='runnerId')
df_runner_market_only_win = df_runner_market[pd.Series(df_runner_market.index.get_level_values('marketId')).apply(lambda x: x in df_market.index and df_market.loc[x]['numberOfWinners'] == 1).values]

# df_runner_market_only_win.groupby('marketId').count().bid.sort_values().plot(use_index=False)
# drop everything with bigger than 15 runners
fill_runners = df_runner_market_only_win.groupby('marketId').count().bid.sort_values().describe()['75%']
df_runner_market_only_win.drop(df_runner_market_only_win.groupby('marketId').count().bid[lambda x: x > fill_runners].index, level='marketId', inplace=True)
combined_market = df_runner_market_only_win.groupby('marketId')[['bid_arr_filled', 'win', 'runner_id']].apply(
    lambda x:( 
        np.concatenate(x['bid_arr_filled']), x['win'].values, x['runner_id'].values
        )
    )
combined_market = pd.DataFrame.from_dict(dict(zip(combined_market.index, combined_market.values)), orient='index', columns=['bid_arr', 'win', 'runner_id'])

all_filled = fill_runners*fill_bid_arr_size

# need to pad them with zeros to feed them to the algorithm

combined_market.bid_arr = combined_market.bid_arr.apply(lambda x: np.pad(x, (0, int(all_filled - len(x))), mode='constant', constant_values=0) if all_filled >= len(x) else print('Array too long'))

combined_market.win = combined_market.win.apply(lambda x: np.pad(x.astype(float), (0, int(fill_runners - len(x))), mode='constant', constant_values=0) if fill_runners >= len(x) else print('Array too long'))

combined_market.runner_id = combined_market.runner_id.apply(lambda x: np.pad(x.astype(float), (0, int(fill_runners - len(x))), mode='constant', constant_values=0) if fill_runners >= len(x) else print('Array too long'))

#%% get the ones with the most favourable odds
train_market = combined_market.iloc[:int(len(combined_market)*0.8)]
test_market = combined_market.iloc[int(len(combined_market)*0.8):]
train_value = np.vstack(train_market.bid_arr.values)
train_label = np.vstack(train_market.win.values)
test_value = np.vstack(test_market.bid_arr.values)
test_label = np.vstack(test_market.win.values)

# runner_list = df_runner_market.runnerId.unique()

#%% use Kneighbors regressor


print('Start training models...')
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
pred_winner = np.argmax(xgb_preds, axis=1)
act_winner = np.argmax(test_label, axis=1)
print('number of correct prediction is: ', sum(act_winner==pred_winner))
mse_pred = ((xgb_preds - test_label)**2).mean()
print(' xgb pred mse ', mse_pred)


#%% calculating winnings
print('Start calculating winnings...')
test_market['marketId'] = test_market.index
test_market['pred_win_runner'] = [test_market.iloc[i]['runner_id'][w] for i, w in enumerate(pred_winner)]
df_runner_market['market_id'] = df_runner_market.index.get_level_values('marketId')
test_market_merged = test_market.merge(df_runner_market, left_on=['marketId', 'pred_win_runner'], right_on=['market_id', 'runner_id'])
#%%
all_winning = test_market_merged[['bid_arr_y', 'win_y']].apply(lambda x: x[0][-1] if x[1] else 0, axis=1).to_numpy() - 1
sum_winning = sum(all_winning)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

sum_period = 5
sum_winning = [sum(all_winning[current: current+sum_period]) for current in range(0, len(all_winning), sum_period)]

print('total winnings: ', np.sum(sum_winning) - len(sum_winning), ' over ', len(all_winning), ' races. ')
plt.figure()
plt.title('Moving average')
plt.plot(moving_average(all_winning - 1, 5))
plt.savefig('Moving average plot')

plt.figure()
plt.title('Sum of wining')
plt.plot(np.array(sum_winning) - 5, '*')
plt.savefig('sum winning plot')

# print(winning)

# %%
