# The Birch Example Programs

Demonstration and test programs for the Birch probabilistic programming language.


## License

Birch is open source software.

It is licensed under the Apache License, Version 2.0 (the "License"); you may not use it except in compliance with the License. You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>.


## Installation

To build and install, use:

    birch build
    birch install

Then, to run an example, use `birch` followed by the name of the program.

There are several different programs that can be found in the `bi/` subdirectory. Many of these are meant to be read, but do little interesting when run. Some more interesting programs to run demonstrate the delayed sampling mechanism of Birch:

    birch delay_triplet
    birch delay_canonical
    birch delay_iid
    birch delay_spike_and_slab
    birch delay_kalman

Two more interesting examples are models on which the generic `sample` program from the Birch standard library can be run. The first is a linear-Gaussian state-space model, for which Birch will run a Kalman filter (make sure to create the `output/` folder if it does not exist):

    birch sample \
        --model LinearGaussianSSM \
        --input-file input/LinearGaussianSSM.json \
        --output-file output/LinearGaussianSSM.json \
        --ncheckpoints 10

and a mixed linear-nonlinear Gaussian state-space model, for which Birch will run a particle filter:

    birch sample \
        --model LinearNonlinearSSM \
        --input-file input/LinearNonlinearSSM.json \
        --output-file output/LinearNonlinearSSM.json \
        --nparticles 256 \
        --ncheckpoints 50

You can find these models in the `bi/model` subdirectory. Each is implemented as a separate class. The (simulated) data sets used with them are in the `input/` subdirectory. After running, a single posterior sample is output to the `output/` subdirectory.


## Version history

### v0.0.0

* Pre-release.
