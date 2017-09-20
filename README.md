# Birch

Birch is a compiled, imperative, object-oriented, and probabilistic programming language. The latter is its primary research concern. The Birch compiler uses C++ as a target language.

Birch is in development and is not ready for serious work without the expert guidance of its developers. This repository has been opened to satisfy the curious and impatient. Mileage will vary. Contributions are welcome.

The main developer of Birch is [Lawrence Murray](http://www.indii.org/research) at Uppsala University. It is financially supported by the Swedish Foundation for Strategic Research (SSF) via the project *ASSEMBLE*.

An early paper on Birch, specifically the *delayed sampling* mechanism that it uses for partial analytical evaluations of probabilistic models, can be found here:

  * L.M. Murray, D. Lundén, J. Kudlicka, D. Broman and T.B. Schön (2017). *Delayed Sampling and Automatic Rao--Blackwellization of Probabilistic Programs*. Online at <https://arxiv.org/abs/1708.07787>.

## License

Birch is open source software.

It is licensed under the Apache License, Version 2.0 (the "License"); you may not use it except in compliance with the License. You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>.

## Installation

### Installing from Git

If you have acquired Birch directly from its Git repository, first run the following command from within the `Birch` directory:

    ./autogen.sh
    
then follow the instructions for *Installing from source*.

### Installing from source

Birch requires the Boost libraries and Eigen linear algebra library. These should be installed first.

To build and install, run the following from within the `Birch` directory:

    ./configure
    make
    make install


### Installing the standard library

You will also want to install the standard library. It is in a separate `Birch.Standard` repository. To build and install, run the following from within the `Birch.Standard` directory:

    birch build
    birch install

### Installing the examples

You may also want to install the example programs. These are in a separate `Birch.Example` repository. To build and install, run the following from within the `Birch.Example` directory:

    birch build
    birch install

Then, to run an example, use:

    birch example

replacing `example` with the name of the example program. See the `DOCS.md` file for programs and their options.


## Documentation

See the `DOCS.md` file.


## Version history

### v0.0.0

* Pre-release.
