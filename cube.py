import numpy as np
import toolz as tz
import operator as op
from timedeco import timeit
import functools as ft


class Cube(object):
    def __init__(self, center, size, n_div):
        self.center = np.asarray(center).astype(np.float32)
        self.size = np.abs(np.float32(size))
        self.n_div = np.abs(np.int32(n_div))

    @property
    def shape(self):
        return (self.n_div for _ in range(3))

    def _get_coordinate_1(self, key):
        dim, key = key
        return self.center[dim] - self.size + 2 * self.size / (
            self.n_div - 1) * key

    def get_coordinate(self, *key):
        return map(self._get_coordinate_1, enumerate(key))

    def contains(self, r):
        c = list(self.center)
        rel = map(lambda t: np.abs(t[0] - t[1]), zip(c, r))
        tf = map(lambda a: a < self.size, rel)
        onezero = map(lambda a: a.astype(np.int64), tf)
        return tz.reduce(lambda x, y: x * y, onezero)


def cube_to_coordinate(cube):
    return np.fromfunction(cube.get_coordinate, cube.shape)


if __name__ == '__main__':
    cube = Cube([0, 0, 0], 2, 4)
    x, y, z = cube_to_coordinate(cube)
    print(z)
