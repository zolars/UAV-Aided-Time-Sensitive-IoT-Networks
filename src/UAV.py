import params

import numpy as np


class UAV:
    """UAV class
    Model UAV and provide functions used to locate it.
    """

    def __init__(self, copy=None):
        if copy is None:
            self.x = 0.0
            self.y = 0.0
            self.records = [(0.0, (0.0, 0.0))]
            self.t_limit = params.t_limit
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
        distance = np.sqrt((self.x - sensor.x)**2 +
                           (self.y - sensor.y)**2) - params.s
        distance = distance if distance > 0.01 else 0.0
        if distance == 0:
            t = params.period - self.records[-1][0] % params.period - 1
        else:
            t = distance / params.v

        if self.records[-1][0] + t > params.max_time:
            return False
        else:
            if distance != 0.0:
                _x, _y = self.x + (sensor.x - self.x) * \
                    (distance / np.sqrt((self.x - sensor.x)**2 + (self.y - sensor.y)**2)), \
                    self.y + (sensor.y - self.y) * \
                    (distance / np.sqrt((self.x - sensor.x)**2 + (self.y - sensor.y)**2))

                t_back = np.sqrt(_x**2 + _y**2) / params.v
                if self.t_limit - t - t_back < 0:
                    return self.back() + self.fly_to(sensor)

                self.x, self.y = _x, _y

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
        t = distance / params.v
        self.x = 0
        self.y = 0
        self.records.append(
            (self.records[-1][0] + t, (self.x, self.y)))

        self.t_limit = params.t_limit

        return t
