import ase
from multipledispatch import dispatch
import toolz as tz
import math
import functools as ft
import numpy as np
from numpy.linalg import norm
from nlm import nlm
from cube import Cube
from cube import cube_to_coordinate


class Atom(object):
    @dispatch(ase.Atom)
    def __init__(self, atom):
        self.n, self.l, self.m = nlm(atom.number)
        self.r = atom.position.astype(np.float32)
        self.symbol = atom.symbol

    @property
    def nlm(self):
        return (self.n, self.l, self.m)

    def __repr__(self):
        return 'Atom[{},{},{},{}]'.format(self.n, self.l, self.m, self.r)

    def to_ase(self):
        return ase.Atom(self.symbol, self.r)


class Atoms(list):
    @dispatch(ase.Atoms)
    def __init__(self, atoms):
        super().__init__(tz.map(Atom, atoms))

    def to_ase(self):
        atoms = list(map(lambda a: a.to_ase(), self))
        return ase.Atoms(atoms)


class Hand(object):
    def __init__(self, r):
        self.r = np.asarray(r).astype(np.float32)


def coordinate_to_rel(coordinate, obj):
    return map(lambda a: a[0] - a[1], zip(coordinate, tuple(obj.r)))


def rel_to_sqr_distance(rel):
    return map(lambda a: a**2, rel)


def sqr_distance_to_gauss(sqr, alpha):
    return np.exp(-alpha * sum(sqr))


def cube_to_to_gauss(cube):
    coor = list(cube_to_coordinate(cube))

    def atom_to_gauss(atom):
        rel = coordinate_to_rel(coor, atom)
        sqr = rel_to_sqr_distance(rel)
        gau = sqr_distance_to_gauss(sqr, 1)
        return np.array([gau * atom.n, gau * atom.l, gau * atom.m])

    def hand_to_gauss(hand):
        rel = coordinate_to_rel(coor, hand)
        sqr = rel_to_sqr_distance(rel)
        gau = sqr_distance_to_gauss(sqr, 1)
        return gau

    return atom_to_gauss, hand_to_gauss


def in_small_cube(cube, small):
    coor = cube_to_coordinate(cube)
    return small.contains(coor)


class Field(object):
    def __init__(self, atoms):
        self.n_div = 16
        self.n_sight = 3
        self.atoms = Atoms(atoms)
        self.hand = Hand([0, 0, 0])
        self.cubes = [Cube([0, 0, 0], 4, self.n_div)
                      for _ in range(self.n_sight)]
        self.cubes.append(Cube([0, 0, 0], 0, 0))
        self.cubes[1].size *= 0.75
        self.cubes[2].size *= 0.50

    def selected_atom(self):
        dis = tz.map(lambda atom: norm(atom.r - self.hand.r), self.atoms)
        dis = np.array(list(dis))
        i = np.argmin(dis)
        return self.atoms[i]

    def make_sight(self, i):
        atom_to_gauss, hand_to_gauss = cube_to_to_gauss(self.cubes[i])
        atomic = sum(atom_to_gauss(atom) for atom in self.atoms)
        handic = hand_to_gauss(self.hand)
        contai = in_small_cube(self.cubes[i], self.cubes[i + 1])
        return np.concatenate([atomic, (handic + contai)[None, :]]).astype(
            np.float32)

    def make_sights(self):
        return [self.make_sight(i) for i in range(self.n_sight)]

    def pull(self):
        selected = self.selected_atom()
        rel = self.hand.r - selected.r
        selected.r += rel * 0.1

    def calc(self):
        print("calc not implemented")
        atoms = self.atoms.to_ase()
        return 0


if __name__ == '__main__':
    spam = ase.Atoms('HH', [[0, 0, 0], [2, 2, 2]])
    ham = Field(spam)
    egg = ham.make_sight(0)
    np.savetxt('a.vasp',egg[3].flatten())
    print("end")
