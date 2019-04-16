import ase
import numpy as np
from collections import namedtuple
import toolz as tz
import math
from multipledispatch import dispatch
import functools as ft
import types
from collections import Iterable


def calculate(atoms):
    # return energy
    pass


def selection_table():
    selection_table = np.zeros((10, 10))
    for n in range(selection_table.shape[0]):
        for l in range(selection_table.shape[1]):
            if n >= l:
                selection_table[n, l] = 2 * (2 * l + 1)
    return selection_table.astype(np.int64)


table = selection_table()


def nlm(atomic_number):
    for i in range(10):
        for j in range(i, -1, -1):
            if table[i-j, j] < atomic_number:
                atomic_number -= table[i-j, j]
            else:
                return i-j+1, j+1, atomic_number
    assert(False)


@dispatch(ase.Atom)
def nlmr(atom):
    nlmr = namedtuple('nlmr', ['n', 'l', 'm', 'r'])
    return nlmr(*nlm(atom.number), np.asarray(atom.position))


@dispatch(ase.Atoms)
def nlmr(atoms):
    return list(map(nlmr, atoms))


@dispatch(np.ndarray, np.ndarray)
def gauss(r, o):
    return math.exp(-(r - o) @ (r - o))


class Cube(object):
    def __init__(self, center, size, n_div):
        self.center = center
        self.size = size
        self.n_div = np.asarray(n_div).astype(np.int64)

    @property
    def center(self):
        return self._center

    @property
    def size(self):
        return self._size

    @center.setter
    def center(self, center):
        self._center = np.asarray(center)

    @size.setter
    def size(self, size):
        self._size = np.abs(np.asarray(size))

    def __contains__(self, point):
        point = np.asarray(point)
        rel = point - self.center
        for i in range(3):
            if abs(rel[i]) > self.size[i]:
                return False
        return True

    def __getitem__(self, key):
        k = np.asarray(key)
        if np.any(self.n_div <= k):
            raise KeyError()
        return (self.center - self.size) + 2 * self.size / (self.n_div - 1) * k

    def __iter__(self):
        return self._iter()

    def _iter(self):
        key = [0, 0, 0]
        while True:
            yield tuple(key), self[key]
            if key[2] < self.n_div[2] - 1:
                key[2] += 1
            else:
                key[2] = 0
                if key[1] < self.n_div[1] - 1:
                    key[1] += 1
                else:
                    key[1] = 0
                    key[0] += 1
                    if key[0] == self.n_div[0] - 1:
                        raise StopIteration()


class Simulater(object):
    def __init__(self, atoms, select=0, grid=(32, 32, 32)):
        self.atoms = atoms
        self.select = select
        self.grid = (*grid, 4)
        self.eyes = [np.zeros(self.grid), np.zeros(self.grid)]
        self.cubes = [Cube((-5, -5, -5), (5, 5, 5), grid), Cube((-2, -2, -2), (2, 2, 2), grid)]

    def make_grid(self):
        self.eyes[0] = np.zeros(self.grid)
        for key, o in self.cubes[0]:
            for i in range(4):
                key = (*key, i)
                for atom in nlmr(self.atoms):
                    if i < 3:
                        self.eyes[0][key] += gauss(atom.r, o) * atom[i]
                    else:
                        if o in self.eyes[1]:
                            self.eyes[0][key] = 1
                        else:
                            self.eyes[0][key] = 0




a = ase.Atoms('HB', [[0,0,0],[1,1,1]])
spam = Simulater(a)
spam.make_grid()
print(spam.cubes[0])
