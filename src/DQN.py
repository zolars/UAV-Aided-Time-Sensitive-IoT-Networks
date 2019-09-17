import params
from utils import draw, cost, generateMap, observe

from UAV import UAV
from Sensor import Sensor
from DQN_brain import DeepQNetwork

import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def run(episode,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=200,
        memory_size=2000):

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
    print('  algorithm:\t\tDQN')
    print('  episode:\t\t', episode)
    print('  learning_rate:\t', learning_rate)
    print('  reward_decay:\t\t', reward_decay)
    print('  e_greedy:\t\t', e_greedy)
    print('  replace_target_iter:\t', replace_target_iter)
    print('  memory_size:\t\t', memory_size)
    print('-----------------------------------------------')

    RL = DeepQNetwork(
        params.sensors_amount,
        params.sensors_amount,
        learning_rate=learning_rate,
        reward_decay=reward_decay,
        e_greedy=e_greedy,
        replace_target_iter=replace_target_iter,
        memory_size=memory_size,
        # output_graph=True
    )

    costs = []
    best_uav, best_result, best_cost = None, None, float('inf')
    step = 0
    for _ in tqdm(range(episode)):
        # initial observation
        sensors, uav = generateMap()
        observation = observe(uav, sensors)
        np.random.seed()

        previous_cost = cost(uav, sensors)
        pass
        while True:
            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            done = uav.fly_to(sensors[action]) is False
            _cost = cost(uav, sensors)
            _observation, reward = observe(
                uav, sensors), (previous_cost - _cost) * 100
            previous_cost = _cost

            # RL learn from this transition
            RL.store_transition(observation, action, reward, _observation)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

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
            step += 1

    # output results
    print('Max time', params.max_time, 'Final time:', best_uav.records[-1][0])
    print('Best cost:', best_cost)

    # show costs plot
    # x, y = list(range(episode)), costs
    # plt.plot(x, y, color='red')
    # plt.show()
    # RL.plot_cost()

    # draw(best_uav, sensors, details=True)
    # draw(best_uav, sensors, details=False)
    del (RL)
    return best_result


if __name__ == "__main__":
    best_result = run(episode=100)
    with open('./out/DQN_{:%m-%d-%H-%M-%S}.json'.format(params.time),
              "w+") as f:
        f.write(json.dumps(best_result))
        f.close()
