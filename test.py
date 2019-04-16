#! /Users/hayashiakihide/.pyenv/shims/python
import theano.tensor as T
import theano
import numpy as np
import ase
from ase.io import xyz
from ase.calculators import gaussian
from numba import jit
import cube

# mol = ase.Atoms('HH', [[0,0,0],[1,1,1]])
# g = gaussian.Gaussian(atoms=mol, method='UB3LYP')
# g.calculate(mol, properties=['energy'])


# @jit(nopython=True)
# def f(x, y, z):
#     # print(type(x))
#     print("x")
#     print(x)
#     print("y")
#     print(y)
#     print("z")
#     print(z)
#     # print(type(y))
#     # print(y)
#     c = cube.Cube([0, 0, 0], 2, 4)
#     print(c.get_coordinate(x, y, z))
#     return x * y
#
# a = np.fromfunction(f, (4, 4, 4))
# print("result")
# print(a)

a = ase.Atoms('HH', [[0,0,0], [1,1,1]])
with open("test.xyz", 'w') as f:
    xyz.write_xyz(f, a)
