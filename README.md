name: Standard
version: 0.0.0
description: The Birch standard library.
---

# Birch

Birch is a compiled, imperative, object-oriented, and probabilistic programming language. The latter is its primary research concern. The Birch compiler uses C++ as a target language.

## Installation

### Installing from Git

If you have acquired Birch directly from its Git repository, first run the following command from within the `Birch` directory:

    ./autogen.sh
    
then follow the instructions for *Installing from source*.

### Installing from source

Birch requires the Boost libraries, Eigen linear algebra library and Boehm garbage collector (`libgc`). These should be installed first.

To build and install, run the following from within the `Birch` directory:

    ./configure
    make
    make install
    
This installs three components:

  1. `birch` (the driver program),
  2. `birchc` (the compiler program), and
  3. `libbirch.*` and associated `bi/*.hpp` header files (the compiler library).

Typically, only the first of these is used directly. It provides a friendly wrapper for building and running Birch code, calling the compiler program, and linking in the compiler library, where appropriate. It is usually unnecessary to become familiar with the latter two.

### Installing the standard library

You will also want to install the standard library. It is in a separate `Birch.Standard` repository. To build and install, run the following from within the `Birch.Standard` directory:

    birch build
    birch install

### Installing the examples

You may also want to install the example programs. These are in a separate `Birch.Example` repository. To build, run the following from within the `Birch.Example` directory:

    birch build

Then, to run an example, use:

    birch example

replacing `example` with the name of the example program. See the `DOCS.md` file for programs and their options.


## Documentation

See the `DOCS.md` file.


## Version history

### v0.0.0

* Pre-release.
