# Birch

Birch is a probabilistic programming language featuring automatic
marginalization, automatic conditioning, automatic differentiation, and
inference algorithms based on Sequential Monte Carlo (SMC). The Birch language
transpiles to C++.

See [https://birch-lang.org](https://birch-lang.org) for a gentle introduction, and
[https://docs.birch-lang.org](https://docs.birch-lang.org) for reference documentation.

[![lawmurray](https://circleci.com/gh/lawmurray/Birch.svg?style=shield)](https://circleci.com/gh/lawmurray/Birch) [![codecov](https://codecov.io/gh/lawmurray/Birch/graph/badge.svg)](https://codecov.io/gh/lawmurray/Birch)


## License

Birch is open source software. It is licensed under the Apache License,
Version 2.0 (the "License"); you may not use it except in compliance with the
License. You may obtain a copy of the License at
<http://www.apache.org/licenses/LICENSE-2.0>.


## Getting started

Binary packages may be available for your system, see [the website](https://birch-lang.org/getting-started/). If not, or if you have special requirements, you can install Birch from source. This requires:

  * GNU autoconf, automake, libtool, flex, and bison
  * [LibYAML](https://pyyaml.org/wiki/LibYAML)
  * [Eigen](https://eigen.tuxfamily.org)

The following is optional but recommended for significant performance
improvements, and will be linked in automatically if found:

  * [jemalloc](http://jemalloc.net/)

All Birch sources are in the same repository. Clone it:

    git clone -b stable https://github.com/lawmurray/Birch.git

and change to the `Birch` directory:

    cd Birch

Then proceed as follows. Note special instructions for Mac in step 2. In
addition, on Mac, you can typically omit `sudo` from these commands.

1. Install MemBirch by running, from within the `membirch/` directory:
   ```
   ./bootstrap
   ./configure
   make
   sudo make install
   ```

2. Install NumBirch by running, from within the `numbirch/` directory:
   ```
   ./bootstrap
   ./configure
   make
   sudo make install
   ```

3. Install Birch by running, from within the `birch/` directory:
   ```
   ./bootstrap
   ./configure
   make
   sudo make install
   ```

4. Install the Birch standard library by running, from within the
   `libraries/Standard/` directory:
   ```
   birch build
   sudo birch install
   ```

This constitutes a basic install. You can inspect the different components for
advanced options, such as disabling assertions to improve performance, or
building the CUDA backend for NumBirch. You may also like to install other
packages in the `libraries/` directory. It is not usual to install the
packages in the `examples/` directory, although you may like to build and run
these locally for learning purposes.
