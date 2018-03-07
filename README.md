# Vector-borne disease models

This package implements a number of vector-borne disease models in Birch. They are useful for modelling epidemics of mosquito-borne diseases such as Zika and dengue. The original model is described in Funk et al. (2016), and the particular model implemented here in Murray et al. (2018).

The implementation is based on the [original implementation](https://github.com/sbfnk/vbd) by Sebastian Funk, which was in [LibBi](http://www.libbi.org). The model has been adapted from continuous-time and state to discrete-time and state.


## Installation

To build, use:

    birch build
    
To install system wide, use:

    birch install

To run, use, e.g.:

    birch sample \
        --model YapModel \
        --ncheckpoints 197 \
        --nparticles 1024 \
        --input-file input/yap_dengue.json \
        --output-file output/yap_dengue.json


## References

  * S. Funk, A.J. Kucharski, A. Camacho, R.M. Eggo, L. Yakob, L.M. Murray, and W.J. Edmunds (2016). [Comparative analysis of dengue and Zika outbreaks reveals differences by setting and virus](http://dx.doi.org/10.1101/043265). PLOS Neglected Tropical Diseases **10**:1-16.

  * L.M. Murray, D. Lundén, J. Kudlicka, D. Broman, and T.B. Schön (2018). [Delayed Sampling and Automatic Rao--Blackwellization of Probabilistic Programs](https://arxiv.org/abs/1708.07787).
