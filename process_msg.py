import pandas as pd
import datetime


def process_msg(msg, market_to_runner, runner_to_market):
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
                    print('runner ', runner_id, ' doesn\'t appear in the marget def')

        return market_to_runner, runner_to_market
