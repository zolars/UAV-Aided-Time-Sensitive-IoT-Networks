import params
import greedy
import QL
import DQN

import json
import datetime

cost = []
for _ in range(10):

    params.sensors_amount += 5
    params.max_time = params.get_max_time() * params.period

    result = {}
    result['greedy'] = greedy.run()
    result['QL'] = QL.run(episode=100)
    result['DQN'] = DQN.run(episode=100)

    cost.append({
        'greedy': result['greedy']['cost'],
        'QL': result['QL']['cost'],
        'DQN': result['DQN']['cost']
    })

    with open('./out/{:%m-%d-%H-%M-%S}.json'.format(datetime.datetime.now()),
              "w+") as f:
        f.write(json.dumps(result))
        f.close()

    params.priority_range += 2

with open(
        './out/sensors_amount_{:%m-%d-%H-%M-%S}.json'.format(
            datetime.datetime.now()), "w+") as f:
    f.write(json.dumps(cost))
    f.close()

print(cost)