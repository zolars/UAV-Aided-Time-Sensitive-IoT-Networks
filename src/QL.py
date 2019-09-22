import params
from utils import draw, cost, generateMap

from UAV import UAV
from Sensor import Sensor
from QL_brain import QTable

import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def run(episode, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):

    print('------------------Environment------------------')
    print('  length_range:\t\t', params.length_range)
    print('  priority_range:\t', params.priority_range)
    print('  sensors_amount:\t', params.sensors_amount)
    print('  s:\t\t\t', params.s)
    print('  v:\t\t\t', params.v)
    print('  period:\t\t', params.period)
    print('  t_limit:\t\t', params.t_limit)
    print('  max_time:\t\t', params.max_time)
    print('  Random seed:\t\t', params.seed)
    print('--------------------Method---------------------')
    print('  algorithm:\t\tQL')
    print('  episode:\t\t', episode)
    print('  learning_rate:\t', learning_rate)
    print('  reward_decay:\t\t', reward_decay)
    print('  e_greedy:\t\t', e_greedy)
    print('-----------------------------------------------')

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
            _observation = (str(action) + '_' + observation)
            reward = (previous_cost - _cost) * 100
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
        del uav

    # output results
    print('Max time', params.max_time, 'Final time:', best_uav.records[-1][0])
    print('Best cost:', best_cost)

    # print('Q_table:\n', RL.q_table)

    # x, y = list(range(episode)), costs
    # plt.plot(x, y, color='red')
    # plt.show()

    # draw(best_uav, sensors, details=True)
    # draw(best_uav, sensors, details=False)

    return best_result


if __name__ == "__main__":
    best_result = run(episode=100)
    with open('./out/QL_{:%m-%d-%H-%M-%S}.json'.format(params.time),
              "w+") as f:
        f.write(json.dumps(best_result))
        f.close()
