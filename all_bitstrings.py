import numpy as np
import jax.numpy as jnp
import jax
import itertools
from functools import partial

jax.config.update("jax_enable_x64", True)

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

@partial(jax.jit, static_argnums=(1,))
def cycle_step(counter_period, _):
    counter, period = counter_period
    counter = jax.lax.cond(counter >= period-1, lambda: -period, lambda: counter+1)
    bit = jax.lax.cond(
        counter >= 0,
        lambda: jnp.array(0, dtype=jnp.int64),
        lambda: jnp.array(1, dtype=jnp.int64)
    )
    return (counter, period), bit

def build_column(index, size):
    counter_period = (-1, 2 ** index)
    return jax.lax.scan(cycle_step, counter_period, None, 2**size)[1]

def all_bitstrings_jax(size):
    return jax.vmap(lambda i: build_column(i, size), out_axes=1)(jnp.arange(size-1, -1, -1))
