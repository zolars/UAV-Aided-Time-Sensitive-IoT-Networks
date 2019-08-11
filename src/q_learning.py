import params
from utils import draw, cost, generateMap

from UAV import UAV
from Sensor import Sensor
from RL_brain import QTable

import numpy as np
from tqdm import tqdm


def q_learning(episode=100, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
    RL = QTable(actions=list(range(params.sensors_amount)),
                learning_rate=learning_rate,
                reward_decay=reward_decay,
                e_greedy=e_greedy)
    for _ in tqdm(range(episode)):
        # initial observation
        observation = -1
        sensors, uav = generateMap()
        np.random.seed()

        while True:
            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            done = not uav.fly_to(sensors[action])
            observation_, reward = action, 100 / cost(uav, sensors)

            # RL learn from this transition
            RL.learn(observation, action, reward, observation_)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # output results
    print('Max time', params.max_time, 'Final time:', uav.records[-1][0])
    cost(uav, sensors, output=True)
    draw(uav, sensors, details=True)
    draw(uav, sensors, details=False)


if __name__ == "__main__":
    q_learning(episode=100,
               learning_rate=1,
               reward_decay=0.9,
               e_greedy=0.9)
