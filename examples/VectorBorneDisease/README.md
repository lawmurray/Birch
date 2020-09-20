# Vector-borne disease package

This package implements a number of vector-borne disease models in Birch. They are useful for modelling epidemics of mosquito-borne diseases such as Zika and dengue. The original model is described in [Funk et al. (2016)](#references), and the particular model implemented here in [Murray et al. (2018)](#references).

The implementation is based on the [original implementation](https://github.com/sbfnk/vbd) by Sebastian Funk, which was in [LibBi](http://www.libbi.org). The model has been adapted from continuous-time and state to discrete-time and state.


## License

This package is open source software.

It is licensed under the Apache License, Version 2.0 (the "License"); you may not use it except in compliance with the License. You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>.


## Getting started

To build, use:

    birch build

To run, use:

    ./run.sh


## References

  * S. Funk, A.J. Kucharski, A. Camacho, R.M. Eggo, L. Yakob, L.M. Murray, and W.J. Edmunds (2016). [Comparative analysis of dengue and Zika outbreaks reveals differences by setting and virus](http://dx.doi.org/10.1101/043265). PLOS Neglected Tropical Diseases **10**:1-16.

  * L.M. Murray, D. Lundén, J. Kudlicka, D. Broman, and T.B. Schön (2018). [Delayed Sampling and Automatic Rao--Blackwellization of Probabilistic Programs](https://arxiv.org/abs/1708.07787).
