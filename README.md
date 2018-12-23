# Multiple object tracking

A simple multiple object tracking demonstration in the Birch probabilistic programming language. This demonstrates the use of a universal probabilistic programming language for inference on a model without fixed dimension (the number of objects is unknown). Data is simulated from the model and then filtered using a particle filter, within which the delayed sampling heuristic (Murray *et al.* 2018) automatically yields a Kalman filter for the tracking of each object. It is used as an example in Murray & Schön (2018), in which further details are available.


## License

This package is licensed under the Apache License, Version 2.0 (the "License"); you may not use it except in compliance with the License. You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>.


## Getting started

This package requires the `Birch.Cairo` package, which should be installed first.

To build:

    birch build
    
To install:

    birch install

To run:

    birch run

Inspect the `run` program for an example of the functionality available.


## References

  1. L.M. Murray and T.B. Schön (2018). [Automated learning with a probabilistic programming language: Birch](https://dx.doi.org/10.1016/j.arcontrol.2018.10.013). *Annual Reviews in Control* **46**:29--43. [[arxiv]](https://arxiv.org/abs/1810.01539)

  2. L.M. Murray, D. Lundén, J. Kudlicka, D. Broman and T.B. Schön (2018). [Delayed Sampling and Automatic Rao&ndash;Blackwellization of Probabilistic Programs](https://arxiv.org/abs/1708.07787). *Proceedings of the 21st International Conference on Artificial Intelligence and Statistics (AISTATS)*.