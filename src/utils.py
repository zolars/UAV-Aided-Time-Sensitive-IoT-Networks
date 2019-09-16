import params
from UAV import UAV
from Sensor import Sensor

import json
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection


def status(uav, sensors):
    return False


def cost(uav, sensors, details=False, output=False):
    sensors_result = []
    cost = 0
    for sensor in sensors:
        epsilon = set()
        for record in sensor.records:
            epsilon.add(record // (params.period * sensor.p))
        epsilon = params.max_time / (params.period * sensor.p) - len(epsilon)

        if epsilon != 0:
            cost += epsilon / sensor.p
        if output or details:
            console = dict()
            console['x'] = sensor.x
            console['y'] = sensor.y
            console['p'] = sensor.p
            console['epsilon'] = epsilon
            console['cost'] = epsilon * 1 / sensor.p
            console['records'] = sensor.records
            sensors_result.append(console)

    result = {
        'params': {
            'length_range': params.length_range,
            'priority_range': params.priority_range,
            'sensors_amount': params.sensors_amount,
            's': params.s,
            'v': params.v,
            'period': params.period,
            't_limit': params.t_limit,
            'max_time': params.max_time,
            'seed': params.seed,
        },
        'cost': cost,
        'uav_result': uav.records,
        'sensors_result': sensors_result
    }

    if output:

        with open('./out/{:%m-%d-%H-%M-%S}.json'.format(params.time),
                  "w+") as f:
            f.write(json.dumps(result))
            f.close()
        print('Total cost:', cost)
    if details:
        return result

    return cost


def draw(uav, sensors, details=False):
    plt.style.use('classic')

    fig = plt.figure(0, facecolor='lightgray', dpi=100)
    ax = fig.add_subplot(111)

    plt.axis('equal')
    ax.plot([0, 0], [0, params.length_range], c='#000000')
    ax.plot([0, params.length_range], [0, 0], c='#000000')
    ax.plot([params.length_range, params.length_range],
            [0, params.length_range],
            c='#000000')
    ax.plot([0, params.length_range],
            [params.length_range, params.length_range],
            c='#000000')

    patches = []
    colors = []
    for sensor in sensors:
        circle = Circle(xy=(sensor.x, sensor.y), radius=params.s)
        patches.append(circle)
        colors.append(1 / params.priority_range * sensor.p * 100)
        ax.scatter(sensor.x, sensor.y, c='b', marker='^')

    p = PatchCollection(patches, cmap=cm.YlOrRd, alpha=0.4)
    ax.add_collection(p)
    p.set_array(np.array(colors))
    plt.colorbar(p)

    previous_x, previous_y = 0, 0
    tour = 0
    for record in uav.records:
        current_x, current_y = record[1][0], record[1][1]
        ax.scatter(current_x, current_y, c='C' + str(tour), alpha=1)
        ax.annotate("",
                    xy=(current_x, current_y),
                    xytext=(previous_x, previous_y),
                    arrowprops={
                        'arrowstyle': "->",
                        'color': 'C' + str(tour)
                    })
        if (current_x, current_y) == (0.0, 0.0):
            tour += 1
            if details:
                if tour > 1:
                    fig.savefig('./out/{:%m-%d-%H-%M-%S}_{}.png'.format(
                        params.time, tour - 1))
                plt.close('all')
                fig = plt.figure(tour, facecolor='lightgray', dpi=100)
                ax = fig.add_subplot(111)
                plt.axis('equal')
                ax.plot([0, 0], [0, params.length_range], c='#000000')
                ax.plot([0, params.length_range], [0, 0], c='#000000')
                ax.plot([params.length_range, params.length_range],
                        [0, params.length_range],
                        c='#000000')
                ax.plot([0, params.length_range],
                        [params.length_range, params.length_range],
                        c='#000000')

                patches = []
                colors = []
                for sensor in sensors:
                    circle = Circle(xy=(sensor.x, sensor.y), radius=params.s)
                    patches.append(circle)
                    colors.append(1 / params.priority_range * sensor.p * 100)
                    ax.scatter(sensor.x, sensor.y, c='b', marker='^')

                p = PatchCollection(patches, cmap=cm.YlOrRd, alpha=0.4)
                ax.add_collection(p)
                p.set_array(np.array(colors))
                plt.colorbar(p)

        previous_x, previous_y = current_x, current_y

    if details:
        fig.savefig('./out/{:%m-%d-%H-%M-%S}_{}.png'.format(params.time, tour))
    else:
        fig.savefig('./out/{:%m-%d-%H-%M-%S}.png'.format(params.time))
        plt.show()


def generateMap():
    sensors = []
    np.random.seed(params.seed)
    for _ in range(params.sensors_amount):
        sensor = Sensor()
        sensors.append(sensor)

    uav = UAV()
    return sensors, uav
