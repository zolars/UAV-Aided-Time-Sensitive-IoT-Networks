from utils import draw, cost, generateMap

import params
import json

from UAV import UAV
from Sensor import Sensor


def run():
    """Greedy Algorithm
    A greedy algorithm is an algorithmic paradigm that follows the problem
    solving heuristic of making the locally optimal choice at each stage with the
    intent of finding a global optimum.
    """

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
    print('  algorithm:\t\tGreedy')
    print('-----------------------------------------------')

    sensors, uav = generateMap()
    done, target_sensor_id = False, None
    c = float('inf')
    while not done:
        for sensor in sensors:
            _uav = UAV(uav)
            done = _uav.fly_to(sensor) is False
            if done:
                break

            _c = 0
            for _sensor in sensors:
                epsilon = set()
                for record in _sensor.records:
                    epsilon.add(record // (params.period * _sensor.p))
                epsilon = params.max_time / (params.period *
                                             _sensor.p) - len(epsilon)
                if epsilon != 0:
                    _c += epsilon * 1 / _sensor.p

            # _c = cost(uav, sensors)

            if _c < c:
                c = _c
                target_sensor_id = sensors.index(sensor)

            try:
                sensor.records.pop()
            except:
                pass

            del _uav

        uav.fly_to(sensors[target_sensor_id])
    best_cost = cost(uav, sensors)
    # output results
    print('Max time', params.max_time, 'Final time:', uav.records[-1][0])
    print('Best cost:', best_cost)

    # draw(uav, sensors, details=True)
    # draw(uav, sensors, details=False)

    return cost(uav, sensors, details=True)


if __name__ == "__main__":
    params.sensors_amount = 5
    best_result = run()
    with open('./out/Greedy_{:%m-%d-%H-%M-%S}.json'.format(params.time),
              "w+") as f:
        f.write(json.dumps(best_result))
        f.close()
