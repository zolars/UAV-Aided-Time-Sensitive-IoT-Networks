import numpy as np
import matplotlib.pyplot as plt


class UAV:
    def __init__(self):
        self.x = 0
        self.y = 0


class Sensor:
    def __init__(self, length_range, priority_range):
        self.x = np.random.randint(0, length_range)
        self.y = np.random.randint(0, length_range)
        self.p = np.random.randint(0, priority_range)
        self.s = length_range / 10

    @staticmethod
    def draw(UAV_coordinate, sensors):
        x, y, p, s = [], [], [], sensors[0].s
        for sensor in sensors:
            x.append(sensor.x)
            y.append(sensor.y)
            p.append(sensor.p)

        plt.figure('Scatter', facecolor='lightgray')
        plt.title('Scatter', fontsize=16)
        plt.grid(linestyle=':', axis='x')

        plt.scatter(x, y,
                    s=s,
                    c=p,
                    cmap='jet_r',
                    alpha=0.5)

        plt.scatter(UAV_coordinate[0], UAV_coordinate[1],
                    c='r', s=20, alpha=0.5)
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    length_range = 1000
    priority_range = 5

    sensors = []
    for _ in range(50):
        sensors.append(Sensor(length_range, priority_range))

    Sensor.draw((0, 0), sensors)
