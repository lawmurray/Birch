# Birch

Birch is a probabilistic programming language featuring automatic
marginalization, automatic conditioning, automatic differentiation, and
inference algorithms based on Sequential Monte Carlo (SMC). The Birch language
transpiles to C++.

See [https://birch.sh](https://birch.sh) for a gentle introduction, and
[https://docs.birch.sh](https://docs.birch.sh) for reference documentation.

[![lawmurray](https://circleci.com/gh/lawmurray/Birch.svg?style=shield)](https://circleci.com/gh/lawmurray/Birch) [![codecov](https://codecov.io/gh/lawmurray/Birch/graph/badge.svg)](https://codecov.io/gh/lawmurray/Birch) [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](https://github.com/lawmurray/Birch/blob/master/CODE_OF_CONDUCT.md) 


## License

Birch is open source software. It is licensed under the Apache License,
Version 2.0 (the "License"); you may not use it except in compliance with the
License. You may obtain a copy of the License at
<http://www.apache.org/licenses/LICENSE-2.0>.


## Getting started

### Linux

Packages are provided for major Linux distributions, including Debian, Ubuntu,
Fedora, CentOS, openSUSE, SUSE Linux Enterprise, Mageia, and Arch. Click
through to the [Open Build
Service](https://software.opensuse.org//download.html?project=home%3Alawmurray%3Abirch&package=birch)
and select your distribution for installation instructions.

For Raspberry Pi OS, head straight to the
[repository](https://download.opensuse.org/repositories/home:/lawmurray:/birch/).
For Alpine Linux, which you may be particularly interested in for installing
Birch in a lightweight container environment, you will need to install [from
source](#from-source), but we do support `musl` for this purpose.

### FreeBSD

You will need to install from source, see below.

### Mac

Install [Homebrew](https://brew.sh) if not already, then install Birch with:

    brew tap lawmurray/birch
    brew install birch

### Windows

Native support is not yet provided, but you can install [Windows Subsystem for
Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10) with a
Linux distribution of your choice, then click through to the [Open Build
Service](https://software.opensuse.org//download.html?project=home%3Alawmurray%3Abirch&package=birch)
and select that distribution for installation instructions.

### From source

If a package is not available for your operating system or you have special
requirements, you can install Birch from source. This requires:

  * GNU autoconf, automake, libtool, flex, and bison
  * [LibYAML](https://pyyaml.org/wiki/LibYAML)
  * [Boost](https://boost.org)
  * [Eigen](https://eigen.tuxfamily.org)

The following are optional, but will be linked in automatically if found (to
improve performance):

  * [jemalloc](http://jemalloc.net/) or
    [tcmalloc](https://gperftools.github.io/gperftools/tcmalloc.html)

All Birch sources are in the same repository. The main branch is considered
stable. Clone it:

    git clone https://github.com/lawmurray/Birch.git

and change to the `Birch` directory:

    cd Birch

Then proceed as follows. Note special instructions for Mac in step 2. In
addition, on Mac, you can typically omit `sudo` from these commands.

1. Install the driver by running, from within the `driver/` directory:

       ./bootstrap
       ./configure
       make
       sudo make install

2. Install LibBirch by running, from within the `libbirch/` directory:

       ./bootstrap
       ./configure --enable-debug --enable-release
       make
       sudo make install

   On Mac, use the following `configure` line instead:

       ./configure --enable-debug --enable-release --disable-openmp CPPFLAGS="-Xpreprocessor -fopenmp"

   Despite the apparently conflicting options, this will *enable*
   multithreading with OpenMP (specifically, it disables the standard check in
   the `configure` script, which fails on Mac, but then adds the required
   flags).

3. Install the standard library by running, from within the
   `libraries/Standard/` directory:

       birch build --enable-debug --enable-release
       sudo birch install

This constitutes a basic install with both *debug* (unoptimized, with
debugging information) and *release* (optimized, without debugging
information) modes. You may also like to install other packages in the
`libraries/` directory. It is not usual to install the packages in the
`examples/` directory, although you may like to build and run these locally
for testing and learning purposes.
