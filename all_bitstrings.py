import numpy as np
import itertools

def all_bitstrings(size):
    bitstrings = np.ndarray((2**size, size), dtype=int)
    for i in range(size):
        bitstrings[:, i] = np.tile(np.repeat(np.array([0, 1]), 2 ** (size-i-1)), 2**i)
    return bitstrings

def all_bitstrings_slow(size):
    bitstrings = np.zeros((2**size, size), dtype=int)
    for i in np.arange(2**size):
        for j, b in enumerate(np.binary_repr(i)[::-1]):
            bitstrings[i, -(j+1)] = int(b)
    return bitstrings

def all_bitstrings_iterator(size):
    return itertools.product([0, 1], repeat=size)
