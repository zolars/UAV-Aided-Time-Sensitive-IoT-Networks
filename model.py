import datetime
import json
import os

from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import numpy as np


def get_max_time():
    l = list(range(1, priority_range + 1))
    b = [i for i in l]

    def all_equal(l):
        start = l[0]
        for key in l[1:]:
            if start != key:
                return False
        return True

    while not all_equal(l):
        min_num = min(l)
        for index, item in enumerate(l):
            if item == min_num:
                l[index] += b[index]
    return l[0]


if not os.path.exists('./out'):
    os.makedirs('./out')

time = datetime.datetime.now()

# Parameters. Please change these number in order to generate different results.
length_range = 1000.0
priority_range = 4
sensors_amount = 10
s = 100.0
v = 10
period = 300
t_limit = 450
max_time = get_max_time() * period

seed = 5

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
print('Result:')


class UAV:
    """UAV class
    Model UAV and provide functions used to locate it.
    """

    def __init__(self, copy=None):
        if copy is None:
            self.x = 0.0
            self.y = 0.0
            self.records = [(0.0, (0.0, 0.0))]
            self.t_limit = t_limit
        else:
            self.x = copy.x
            self.y = copy.y
            self.records = copy.records[:]
            self.t_limit = copy.t_limit

    def fly_to(self, sensor):
        """fly to to specific sensor
        Model the cost for UAV flying to a specific sensor.

        Args:
            sensor: A specific sensor object

        Returns:
            t: The time costed
        """
        distance = np.sqrt((self.x - sensor.x)**2 + (self.y - sensor.y)**2) - s
        distance = distance if distance > 0.01 else 0.0
        if distance == 0:
            t = period - self.records[-1][0] % period - 1
        else:
            t = distance / v

        if self.records[-1][0] + t > max_time:
            return False
        else:
            if distance != 0.0:
                temp_x, temp_y = self.x + (sensor.x - self.x) * \
                    (distance / np.sqrt((self.x - sensor.x)**2 + (self.y - sensor.y)**2)), \
                    self.y + (sensor.y - self.y) * \
                    (distance / np.sqrt((self.x - sensor.x)**2 + (self.y - sensor.y)**2))

                t_back = np.sqrt(temp_x**2 + temp_y**2) / v
                if self.t_limit - t - t_back < 0:
                    return self.back() + self.fly_to(sensor)

                self.x, self.y = temp_x, temp_y

            self.records.append((self.records[-1][0] + t, (self.x, self.y)))
            sensor.records.append(self.records[-1][0])

            self.t_limit -= t

            return t

    def back(self):
        """fly back to (0, 0) in order to get fuel
        Model the cost for UAV flying back to original station.

        Returns:
            t: The time costed
        """
        distance = np.sqrt((self.x - 0)**2 + (self.y - 0)**2)
        t = distance / v
        self.x = 0
        self.y = 0
        self.records.append(
            (self.records[-1][0] + t, (self.x, self.y)))

        self.t_limit = t_limit

        return t


class Sensor:
    """Sensor class
    Model sensors and provide functions used to locate them.
    """

    def __init__(self, length_range, priority_range, seed=None):
        self.x = np.random.randint(0, length_range)
        self.y = np.random.randint(0, length_range)
        self.p = np.random.randint(1, priority_range + 1)
        self.records = []

    def gen(self):
        self.x = 400
        self.y = 400
        self.p = 5


def cost(uav, sensors, output=False):
    sensors_result = []
    cost = 0
    for sensor in sensors:
        epsilon = set()
        for record in sensor.records:
            epsilon.add(record // (period * sensor.p))
        epsilon = max_time / (period * sensor.p) - len(epsilon)

        if epsilon != 0:
            cost += epsilon / sensor.p
        if output:
            console = dict()
            console['x'] = sensor.x
            console['y'] = sensor.y
            console['p'] = sensor.p
            console['epsilon'] = epsilon
            console['cost'] = epsilon * 1 / sensor.p
            console['records'] = sensor.records
            sensors_result.append(console)
            print(console)

    if output:
        result = {
            'params': {
                'length_range': length_range,
                'priority_range': priority_range,
                's': s,
                'v': v,
                'period': period,
                't_limit': t_limit,
                'max_time': max_time,
                'seed': seed,
            },
            'cost': cost,
            'uav_result': uav.records,
            'sensors_result': sensors_result
        }
        with open('./out/{:%m-%d-%H-%M-%S}.json'.format(time), "w+") as f:
            f.write(json.dumps(result))
            f.close()
        print('Total cost:', cost)

    return cost


def draw(uav, sensors, details=False):
    plt.style.use('classic')

    fig = plt.figure('UAV', facecolor='lightgray', dpi=100)
    ax = fig.add_subplot(111)

    if not details:
        plt.axis('equal')
        plt.plot([0, 0], [0, length_range], c='#000000')
        plt.plot([0, length_range], [0, 0], c='#000000')
        plt.plot([length_range, length_range], [0, length_range], c='#000000')
        plt.plot([0, length_range], [length_range, length_range], c='#000000')

        patches = []
        colors = []
        for sensor in sensors:
            circle = Circle(xy=(sensor.x, sensor.y),
                            radius=s)
            patches.append(circle)
            colors.append(1 / priority_range * sensor.p * 100)
            ax.scatter(sensor.x, sensor.y, c='b', marker='^')

        p = PatchCollection(patches, cmap=cm.YlOrRd, alpha=0.4)
        ax.add_collection(p)
        p.set_array(np.array(colors))
        plt.colorbar(p)

    previous_x, previous_y = 0, 0
    tour = 0
    for record in uav.records:
        current_x, current_y = record[1][0], record[1][1]
        ax.scatter(current_x, current_y,
                   c='C' + str(tour), alpha=1)
        ax.annotate("", xy=(current_x, current_y), xytext=(previous_x, previous_y),
                    arrowprops={
            'arrowstyle': "->",
            'color': 'C' + str(tour)
        })
        if (current_x, current_y) == (0.0, 0.0):
            tour += 1
            if details:
                if tour > 1:
                    fig.savefig(
                        './out/{:%m-%d-%H-%M-%S}_{}.png'.format(time, tour - 1))
                plt.close('all')
                fig = plt.figure(tour, facecolor='lightgray', dpi=100)
                ax = fig.add_subplot(111)
                plt.axis('equal')
                ax.plot([0, 0], [0, length_range], c='#000000')
                ax.plot([0, length_range], [0, 0], c='#000000')
                ax.plot([length_range, length_range], [
                    0, length_range], c='#000000')
                ax.plot([0, length_range], [
                    length_range, length_range], c='#000000')

                patches = []
                colors = []
                for sensor in sensors:
                    circle = Circle(xy=(sensor.x, sensor.y),
                                    radius=s)
                    patches.append(circle)
                    colors.append(1 / priority_range * sensor.p * 100)
                    ax.scatter(sensor.x, sensor.y, c='b', marker='^')

                p = PatchCollection(patches, cmap=cm.YlOrRd, alpha=0.4)
                ax.add_collection(p)
                p.set_array(np.array(colors))
                plt.colorbar(p)

        previous_x, previous_y = current_x, current_y

    if not details:
        fig.savefig('./out/{:%m-%d-%H-%M-%S}.png'.format(time))
        plt.show()


def generateMap():
    sensors = []
    np.random.seed(seed)
    for _ in range(sensors_amount):
        sensor = Sensor(length_range, priority_range,
                        seed=np.random.randint(0, 10))
        sensors.append(sensor)

    uav = UAV()
    return sensors, uav


def greedy_err(sensors, uav):
    running, target_sensor = True, None
    while running:
        c = float('inf')
        for sensor in sensors:
            temp_uav = UAV(uav)
            running = temp_uav.fly_to(sensor)
            temp_c = cost(temp_uav, sensors)
            print('Index:', sensors.index(sensor), '\tcost: ', temp_c)

            if len(sensor.records) > 0:
                sensor.records.pop()

            if temp_c <= c:
                c = temp_c
                target_sensor = sensor
        uav.fly_to(target_sensor)
        del(temp_uav)

    cost(uav, sensors, output=True)
    draw(uav, sensors, details=True)
    draw(uav, sensors, details=False)


def genetic(sensors, uav):
    pass


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


def test(sensors, uav):
    uav.fly_to(sensors[1])
    uav.fly_to(sensors[1])
    draw(uav, sensors)
    cost(uav, sensors, output=True)


if __name__ == "__main__":
    sensors, uav = generateMap()
    greedy(sensors, uav)
    # test(sensors, uav)
