from params import *
from utils import draw, cost, generateMap

from UAV import UAV
from Sensor import Sensor


def greedy(sensors, uav):
    """Greedy Algorithm
    A greedy algorithm is an algorithmic paradigm that follows the problem
    solving heuristic of making the locally optimal choice at each stage with the
    intent of finding a global optimum.
    """
    running, target_sensor_id = True, None
    while running:
        c = float('inf')
        for sensor in sensors:
            temp_uav = UAV(uav)
            running = temp_uav.fly_to(sensor)
            if not running:
                break
            temp_c = cost(temp_uav, sensors)
            sensor.records.pop()
            if temp_c < c:
                c = temp_c
                target_sensor_id = sensors.index(sensor)

        uav.fly_to(sensors[target_sensor_id])

    cost(uav, sensors, output=True)
    draw(uav, sensors, details=True)
    draw(uav, sensors, details=False)


if __name__ == "__main__":
    print('Environment:')
    print('  length_range:\t\t', length_range)
    print('  priority_range:\t', priority_range)
    print('  sensors_amount:\t', sensors_amount)
    print('  s:\t\t\t', s)
    print('  v:\t\t\t', v)
    print('  period:\t\t', period)
    print('  t_limit:\t\t', t_limit)
    print('  max_time:\t\t', max_time)
    print('  Random seed:\t\t', seed)
    print('Results:')

    sensors, uav = generateMap()
    greedy(sensors, uav)
    # test(sensors, uav)
    # genetic(sensors, uav)
