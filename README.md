# all-bitstrings
Two functions using NumPy to compute all bit strings of a given size.

Notice the difference in compute time:
```Python
%timeit all_bitstrings(10)
# 94.9 µs ± 491 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```
```Python
%timeit all_bitstrings_slow(10)
# 3.03 ms ± 51.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

[This](https://github.com/frederikwilde/all-bitstrings/blob/main/all_bitstrings.py#L16) neat way to get an iterator was pointed out to me by [@steve_quantum](https://twitter.com/steve_quantum)
```Python
%timeit list(all_bitstrings_iterator(10))
# 33.1 µs ± 88.9 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```

## JIT compiling with JAX
`all_bitstrings_jax` is implemented in such a way that it can be JIT compiled.
For larger sizes this gives a significant speedup.
```Python
%timeit all_bitstrings(24)
# 6.85 s ± 344 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```
```Python
%timeit list(all_bitstrings_iterator(24))
# 9.21 s ± 262 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```
```Python
%timeit all_bitstrings_jax(24).block_until_ready()
# 1.61 s ± 252 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```
However, the JAX version is able to utilize about 200% of CPU (two cores), while the other functions run on strictly one core.
