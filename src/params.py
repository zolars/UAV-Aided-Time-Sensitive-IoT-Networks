import os
import datetime


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
priority_range = 6
sensors_amount = 10
s = 100.0
v = 10
period = 300
t_limit = 450
max_time = get_max_time() * period * 5

seed = 6
