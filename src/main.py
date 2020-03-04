'''
Usage:
conda activate UAV
nohup python ./src/main.py > ./log/main.log $
'''

import params
import greedy
import QL
import DQN

import json
import datetime

cost = []
for _ in range(5):

    # params.sensors_amount += 5
    # params.seed += 5
    params.max_time = params.get_max_time() * params.period

    result = {}
    result['greedy'] = greedy.run()
    result['QL'] = QL.run(episode=100)
    result['DQN'] = DQN.run(episode=100)

    # try:
    #     assert result['greedy']['cost'] > result['DQN']['cost']
    # except:
    #     pass

    cost.append({
        'greedy': result['greedy']['cost'],
        'QL': result['QL']['cost'],
        'DQN': result['DQN']['cost']
    })

    with open('./out/{:%m-%d-%H-%M-%S}.json'.format(datetime.datetime.now()),
              "w+") as f:
        f.write(json.dumps(result))
        f.close()

    params.priority_range += 1

with open(
        './out/priority_range_{:%m-%d-%H-%M-%S}.json'.format(
            datetime.datetime.now()), "w+") as f:
    f.write(json.dumps(cost))
    f.close()

print(cost)