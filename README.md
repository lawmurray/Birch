# Birch

Birch is a probabilistic programming language featuring automatic marginalization, automatic conjugacy, automatic differentiation, and inference algorithms based on Sequential Monte Carlo (SMC). The Birch language transpiles to C++.

See [https://birch.sh](https://birch.sh) for a gentle introduction, and [https://docs.birch.sh](https://docs.birch.sh) for reference documentation.

[![lawmurray](https://circleci.com/gh/lawmurray/Birch.svg?style=shield)](https://circleci.com/gh/lawmurray/Birch) [![codecov](https://codecov.io/gh/lawmurray/Birch/graph/badge.svg)](https://codecov.io/gh/lawmurray/Birch) [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md) 


## License

Birch is open source software. It is licensed under the Apache License, Version 2.0 (the "License"); you may not use it except in compliance with the License. You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>.


## Getting started

### Linux

Packages are provided for major Linux distributions. Click through to the [Open Build Service](https://software.opensuse.org//download.html?project=home%3Alawmurray%3Abirch&package=birch) and select your distribution for installation instructions.

### Mac

Install [Homebrew](https://brew.sh) if not already, then install Birch with:

    brew tap lawmurray/birch
    brew install birch

### Windows

Native support is not yet provided, but you can install [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10) with a Linux distribution of your choice, then click through to the [Open Build Service](https://software.opensuse.org//download.html?project=home%3Alawmurray%3Abirch&package=birch) and select that distribution for installation instructions.

### From source

If a package is not available for your operating system or you have special requirements, you can install from source. This requires:

  * GNU autoconf, automake, libtool, flex, and bison
  * [LibYAML](https://pyyaml.org/wiki/LibYAML)
  * [Boost](https://boost.org)
  * [Eigen](https://eigen.tuxfamily.org)

All Birch sources are in the same repository. The master branch is considered stable. Clone it:

    git clone https://github.com/lawmurray/Birch.git

Install the driver by running, from within the `driver/` directory:

    ./bootstrap
    ./configure
    make
    make install

Install LibBirch by running, from within the `libbirch/` directory:

    ./bootstrap
    ./configure
    make
    make install

Install the standard library by running, from within the `libraries/Standard/` directory:

    birch build
    birch install

This constitutes a minimal install. You may also like to install other packages in the `libraries/` directory. It is not usual to install the packages in the `examples/` directory, although you may like to build and run these for testing or learning purposes.
