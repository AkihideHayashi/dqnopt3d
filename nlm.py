import numpy as np


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

if __name__ == '__main__':
    print(table)
