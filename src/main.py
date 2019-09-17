import params
import greedy
import QL
import DQN

import json

result = {}
result['greedy'] = greedy.run()
result['QL'] = QL.run()
result['DQN'] = DQN.run()

with open('./out/{:%m-%d-%H-%M-%S}.json'.format(params.time), "w+") as f:
    f.write(json.dumps(result))
    f.close()