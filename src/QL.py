import params
from utils import draw, cost, generateMap

from UAV import UAV
from Sensor import Sensor
from QL_brain import QTable

import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def QL(episode=100, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
    RL = QTable(actions=list(range(params.sensors_amount)),
                learning_rate=learning_rate,
                reward_decay=reward_decay,
                e_greedy=e_greedy)

    costs = []
    best_uav, best_result, best_cost = None, None, float('inf')
    for _ in tqdm(range(episode)):
        # initial observation
        observation = str(-1)
        sensors, uav = generateMap()
        np.random.seed()

        previous_cost = cost(uav, sensors)

        while True:
            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            done = uav.fly_to(sensors[action]) is False
            _cost = cost(uav, sensors)
            _observation, reward = (
                str(action) + '_' + str(uav.records[-1][0] // params.period)), (previous_cost - _cost) * 100
            previous_cost = _cost

            # RL learn from this transition
            RL.learn(observation, action, reward, _observation)

            # swap observation
            observation = _observation

            # break while loop when end of this episode
            if done:
                costs.append(_cost)
                if _cost < best_cost:
                    best_result = cost(uav, sensors, details=True)
                    best_cost = _cost
                    best_uav = UAV(uav)
                break

    # output results
    print('Max time', params.max_time, 'Final time:', best_uav.records[-1][0])
    print('Best cost:', best_cost)
    print('Q_table:\n', RL.q_table)

    with open('./out/{:%m-%d-%H-%M-%S}.json'.format(params.time), "w+") as f:
        f.write(json.dumps(best_result))
        f.close()

    x, y = list(range(episode)), costs
    plt.plot(x, y, color='red')
    plt.show()

    draw(best_uav, sensors, details=True)
    draw(best_uav, sensors, details=False)


if __name__ == "__main__":
    QL(episode=100,
       learning_rate=0.1,
       reward_decay=0.9,
       e_greedy=0.9)
