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

    sensors, uav = generateMap()
    greedy(sensors, uav)
