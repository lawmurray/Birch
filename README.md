---
name: VBD
version: 0.0.0
---

# Vector-borne disease models

This package implements a number of vector-borne disease models in Birch, based on the following paper:

Funk, S.; Kucharski, A. J.; Camacho, A.; Eggo, R. M.; Yakob, L.; Murray, L. M. & Edmunds, W. J. [Comparative analysis of dengue and Zika outbreaks reveals differences by setting and virus](http://dx.doi.org/10.1101/043265). PLOS Neglected Tropical Diseases, 2016, 10, 1-16.

The implementations are based on the [original implementations by Sebastian Funk](https://github.com/sbfnk/vbd), which were in [LibBi](http://www.libbi.org). Rather than using continuous time and state variables, the implementation here uses discrete time and space variables, in order to exercise the delayed sampling features of Birch.

The data sets from the original package have also been translated into an appropriate file format for Birch here.


## Installation

To build, use:

    birch build
    
To install system wide, use:

    birch install


## Version history

### v0.0.0

* Initialised project.
