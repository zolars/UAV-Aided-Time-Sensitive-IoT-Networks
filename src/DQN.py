import params
from utils import draw, cost, generateMap, observe

from UAV import UAV
from Sensor import Sensor
from DQN_brain import DeepQNetwork

import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def run(episode=100):

    RL = DeepQNetwork(
        params.sensors_amount,
        params.sensors_amount,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=200,
        memory_size=2000,
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

    with open('./out/{:%m-%d-%H-%M-%S}.json'.format(params.time), "w+") as f:
        f.write(json.dumps(best_result))
        f.close()

    x, y = list(range(episode)), costs
    plt.plot(x, y, color='red')
    plt.show()

    draw(best_uav, sensors, details=True)
    draw(best_uav, sensors, details=False)

    RL.plot_cost()


if __name__ == "__main__":
    run(episode=2000)
