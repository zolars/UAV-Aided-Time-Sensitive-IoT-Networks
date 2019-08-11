import params

import numpy as np


class Sensor:
    """Sensor class
    Model sensors and provide functions used to locate them.
    """

    def __init__(self):
        self.x = np.random.randint(0, params.length_range)
        self.y = np.random.randint(0, params.length_range)
        self.p = np.random.randint(1, params.priority_range + 1)
        self.records = []

    def gen(self):
        self.x = 400
        self.y = 400
        self.p = 5
