# LibBirch C++ Library

LibBirch is a C++ template library intended as a compile target for
probabilistic programming languages (PPLs). It provides basic types and
dynamic memory management (reference counting, cycle collection, lazy deep
copy). This is particularly important for memory efficiency in PPLs using
Sequential Monte Carlo (SMC) and related methods for inference.

LibBirch was developed primarily to support the [Birch](https://birch-lang.org)
probabilistic programming language, but may be useful for other purposes too.


## License

LibBirch is open source software.

It is licensed under the Apache License, Version 2.0 (the "License"); you may
not use it except in compliance with the License. You may obtain a copy of the
License at <http://www.apache.org/licenses/LICENSE-2.0>.


## Installing

To build and install, use:

    ./bootstrap
    ./configure
    make
    make install


## Details

### Reference counting

LibBirch provides reference counting support in combination with lazy deep
copy operations. This was originally described in @ref Murray2020
"Murray (2020)" but the algorithm has since been substantially updated. It
will be properly documented in future.

### Cycle collection

For dealing with reference cycles, weak references can be insufficient, and
user-programmed logic to tear down data structures can be problematic with
lazy deep copy operations: it can end up creating new objects for the
purpose of destroying them. For this reason, LibBirch provides the cycle
collection algorithm after
@ref Bacon2001 "Bacon & Rajan (2001)", with some minor adaptations.

### References

@anchor Murray2020
L.M. Murray (2020). [Lazy object copy as a platform for
population-based probabilistic programming](https://arxiv.org/abs/2001.05293).

@anchor Bacon2001
D.F. Bacon and V.T. Rajan (2001). [Concurrent Cycle Collection in
Reference Counted Systems](https://dx.doi.org/10.1007/3-540-45337-7_12).
*ECOOP 2001 --- Object-Oriented Programming*. 207--235.
