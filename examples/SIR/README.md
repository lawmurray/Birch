# SIR package

Simple SIR (susceptible-infectious-recovered) compartmental model for an infectious disease outbreak.


## License

This package is open source software.

It is licensed under the Apache License, Version 2.0 (the "License"); you may not use it except in compliance with the License. You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>.


## Getting started

To build, use:

    birch build

To run, use:

    birch sample --config config/sir.json


## Details

The model on which this is based is described in
[Murray et. al. (2018)](#references).

The parameter model is given by:

$$\begin{align}
\lambda &\sim \mathrm{Gamma}(2,5) \\
\delta &\sim \mathrm{Beta}(2,2) \\
\gamma &\sim \mathrm{Beta}(2,2),
\end{align}$$

where $\lambda$ is a rate of interaction in the population, $\delta$ the
probability of infection when a susceptible individual interacts with an
infectious individual, and $\gamma$ the daily recovery probability.

The initial model for time $t = 0$ is:

$$\begin{align}
s_0 &= 760 \\
i_0 &= 3 \\
r_0 &= 0.
\end{align}$$

The transition model for time $t$ is:

$$\begin{align}
\tau_t &\sim \mathrm{Binomial}\left(s_{t-1}, 1 - \exp\left(\frac{-\lambda
i_{t-1} }{s_{t-1} + i_{t-1} + r_{t-1}}\right) \right) \\
\Delta i_t &\sim \mathrm{Binomial}(\tau_t, \delta) \\
\Delta r_t &\sim \mathrm{Binomial}(i_{t-1}, \gamma),
\end{align}$$

where $\tau_t$ is the number of interactions between infectious and
susceptible individuals, $\Delta i_t$ the number of newly infected
individuals, and $\Delta r_t$ the number of newly recovered individuals.

Population counts are then updated:

$$\begin{align}
s_t &= s_{t-1} - \Delta i_t \\
i_t &= i_{t-1} + \Delta i_t - \Delta r_t \\
r_t &= r_{t-1} + \Delta r_t.
\end{align}$$


## Acknowledgements

This package contains an influenza data set from [Anonymous (1978)](#references), prepared in JSON format for Birch.


## References

1. Anonymous (1978). Influenza in a boarding school. *British Medical Journal*. **1**:587.

2. L.M. Murray, D. Lundén, J. Kudlicka, D. Broman, and T.B. Schön (2018). [Delayed Sampling and Automatic Rao--Blackwellization of Probabilistic Programs](https://arxiv.org/abs/1708.07787).
