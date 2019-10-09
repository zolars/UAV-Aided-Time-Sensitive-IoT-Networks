import params

import numpy as np


class Sensor:
    """Sensor class
    Model sensors and provide functions used to locate them.
    """
    def __init__(self, x=False, y=False, p=False, records=False):
        if x:
            self.x = x
        else:
            self.x = np.random.randint(0, params.length_range)
        if y:
            self.y = y
        else:
            self.y = np.random.randint(0, params.length_range)
        if p:
            self.p = p
        else:
            self.p = np.random.randint(1, params.priority_range + 1)
        if records:
            self.records = records
        else:
            self.records = []

    def gen(self):
        self.x = 400
        self.y = 400
        self.p = 5
