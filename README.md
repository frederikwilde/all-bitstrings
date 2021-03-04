# all-bitstrings
Two functions using NumPy to compute all bit strings of a given size.

Notice the difference in compute time:
```Python
%timeit all_bitstrings(10)
# gives: 94.9 µs ± 491 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit all_bitstrings_slow(10)
# gives: 3.03 ms ± 51.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

[This](https://github.com/frederikwilde/all-bitstrings/blob/main/all_bitstrings.py#L16) neat way to get an iterator was pointed out to me by [@steve_quantum](https://twitter.com/steve_quantum)
```Python
%timeit list(all_bitstrings_iterator(10))
# gives: 33.1 µs ± 88.9 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```
