# Birch

Birch is a compiled, imperative, object-oriented, and probabilistic programming language. The latter is its primary research concern. The Birch compiler uses C++14 as a target language. It runs on Linux, macOS and Windows.

Birch is in development and is not ready for serious work without the expert guidance of its developers. This repository has been opened to satisfy the curious and impatient. Mileage will vary. Contributions are welcome.

The main developer of Birch is [Lawrence Murray](http://www.indii.org/research) at Uppsala University. It is financially supported by the Swedish Foundation for Strategic Research (SSF) via the project *ASSEMBLE*.

An early paper on Birch, specifically the *delayed sampling* mechanism that it uses for partial analytical evaluations of probabilistic models, can be found here:

  * L.M. Murray, D. Lundén, J. Kudlicka, D. Broman and T.B. Schön (2017). *Delayed Sampling and Automatic Rao--Blackwellization of Probabilistic Programs*. Online at <https://arxiv.org/abs/1708.07787>.

## License

Birch is open source software.

It is licensed under the Apache License, Version 2.0 (the "License"); you may not use it except in compliance with the License. You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>.

## Installing

See below to set up an appropriate system and install dependencies. Once these are installed, Birch itself can be installed as follows.

### Installing from source

If you have acquired Birch directly from its Git repository, first run the following command from within the `Birch` directory:

```sh
./autogen.sh
```

To build and install, run the following from within the `Birch` directory:

```sh
./configure
make
make install
```

### Installing the standard library

You will also want to install the standard library. It is in a separate `Birch.Standard` repository. To build and install, run the following from within the `Birch.Standard` directory:

```sh
birch build
birch install
```

### Installing the examples

You may also want to install the example programs. These are in a separate `Birch.Example` repository. To build and install, run the following from within the `Birch.Example` directory:

```sh
birch build
birch install
```

Then, to run an example, use:

```sh
birch example
```

replacing `example` with the name of the example program. See the `DOCS.md` file for programs and their options.

## Documentation

See the `DOCS.md` file.

## Setting up your system and installing dependencies

Birch requires:

  * GNU autoconf, automake and libtool,
  * Flex and Bison,
  * the Boost libraries, and
  * the Eigen 3 linear algebra library.

These are all widely available through package managers. See the guides below for set up on some major operating systems.

### On Ubuntu Linux

Use:

```sh
apt-get install autoconf libtool flex bison libgc-dev libboost-all-dev libeigen3-dev
```

For other Linux distributions, similar packages will be available.

### On macOS

The recommended package manager is [Homebrew](http://brew.sh). Use:

```sh
brew install autoconf automake libtool flex bison boost eigen bdw-gc
```

Note that Birch needs a newer version of Bison than that provided by macOS. The above command installs a newer version.
    
### On Windows 10

Birch can run through the Bash shell.

First, if you haven't already, activate the developer mode:

1. Go to _Settings_.
2. Go to _Update and security_.
3. Go to _For developer_.
4. Activate the developer mode.
5. Wait for the package configuration.

Then configure the Bash shell:

1. Go to _Control panel_.
2. Go to _Programs and features_.
3. Go to _Turn Windows features on or off_.
4. Check _Windows subsystem for Linux_.
5. Press OK.
6. Restart Windows.
7. Afer the restart open the _Command prompt_ (search for `cmd`).
8. Run the command:
    ```sh
    lxrun /install /y
    ```

9. Open the program _Bash on Ubuntu on Windows_ (search for `Bash`). This is a fully-functional Linux Bash shell with access to the Ubuntu repository. You can also access the file system using the usual Bash commands, noting that the folder `C:\` is called `mnt/c/` here.

10. Update Linux packages:
    ```sh
    apt-get upgrade
    apt-get update
    ```

With the Bash shell now working, follow the instructions for Ubuntu Linux above to install dependencies.


## Version history

### v0.0.0

* Pre-release.
