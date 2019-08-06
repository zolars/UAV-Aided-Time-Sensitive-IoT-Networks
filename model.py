import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def get_max_time():
    L = list(range(1, priority_range + 1))
    b = [i for i in L]
    length = len(L)

    def all_equal(l):
        start = l[0]
        for key in l[1:]:
            if start != key:
                return False
        return True

    while not all_equal(L):
        min_num = min(L)
        for index, item in enumerate(L):
            if item == min_num:
                L[index] += b[index]
    return L[0]


length_range = 1000.0
priority_range = 5
s = 100.0
v = 10.0
period = 100
max_time = get_max_time() * period

print('length_range:\t', length_range)
print('priority_range:\t', priority_range)
print('s:\t\t', s)
print('v:\t\t', v)
print('period:\t\t', period)
print('max_time:\t', max_time)


class UAV:
    """UAV class
    Model UAV and provide functions used to locate it.
    """

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.records = [(0.0, (0.0, 0.0))]

    def fly_to(self, sensor):
        """fly to to specific sensor
        Model the cost for UAV flying to a specific sensor.

        Args:
            sensor: A specific sensor object

        Returns:

        """
        distance = np.sqrt((self.x - sensor.x)**2 + (self.y - sensor.y)**2) - s
        distance = distance if distance > 0.0 else 0.0
        t = distance / v
        if self.records[-1][0] + t <= max_time:
            if distance != 0.0:
                self.x, self.y = self.x + (sensor.x - self.x) * \
                    (distance / np.sqrt((self.x - sensor.x)**2 + (self.y - sensor.y)**2)), \
                    self.y + (sensor.y - self.y) * \
                    (distance / np.sqrt((self.x - sensor.x)**2 + (self.y - sensor.y)**2))

            self.records.append((self.records[-1][0] + t, (self.x, self.y)))
            sensor.records.append(self.records[-1][0] + t)

            return t
        else:
            return None

    def back(self):
        distance = np.sqrt((self.x - 0)**2 + (self.y - 0)**2)
        t = distance / v
        self.x = 0
        self.y = 0
        self.records.append(
            (self.records[-1][0] + t, (self.x, self.y)))
        return t


class Sensor:
    """Sensor class
    Model sensors and provide functions used to locate them.
    """

    def __init__(self, length_range, priority_range):
        self.x = np.random.randint(0, length_range)
        self.y = np.random.randint(0, length_range)
        self.p = np.random.randint(1, priority_range)
        self.records = []

    def gen(self):
        self.x = 400
        self.y = 400
        self.p = 5


def cost(uav, sensors):
    cost = 0
    for sensor in sensors:
        epsilon = set()
        for record in sensor.records:
            epsilon.add(record // (period * sensor.p))
        epsilon = max_time / (period * sensor.p) - len(epsilon)
        print(sensor.p, sensor.records, epsilon)
        cost += epsilon * 1 / sensor.p

    return cost


def draw(uav, sensors):
    fig = plt.figure('Scatter', facecolor='lightgray')
    ax = fig.add_subplot(111)

    for sensor in sensors:
        circle = Circle(xy=(sensor.x, sensor.y),
                        radius=s, alpha=1 / priority_range * sensor.p)
        ax.add_patch(circle)

    perious_x, perious_y = 0, 0
    for record in uav.records:
        current_x, current_y = record[1][0], record[1][1]
        ax.scatter(current_x, current_y,
                   c='r', alpha=0.5)
        ax.annotate("", xy=(current_x, current_y), xytext=(perious_x, perious_y),
                    arrowprops=dict(arrowstyle="->"))
        perious_x, perious_y = current_x, current_y

    plt.axis('equal')
    plt.plot([0, 0], [0, length_range])
    plt.plot([0, length_range], [0, 0])
    plt.plot([length_range, length_range], [0, length_range])
    plt.plot([0, length_range], [length_range, length_range])
    plt.show()


def main():
    sensors = []
    # for _ in range(1):
    sensor = Sensor(length_range, priority_range)
    sensor.gen()
    sensors.append(sensor)

    uav = UAV()
    t = 0
    while True:
        try:
            t += uav.fly_to(sensors[0])
            t += uav.back()
        except:
            break

    draw(uav, sensors)
    c = cost(uav, sensors)
    print('cost: ', c)


if __name__ == "__main__":
    main()
