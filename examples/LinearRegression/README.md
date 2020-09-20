# LinearRegression package

Bayesian linear regression model with conjugate normal-inverse-gamma prior.

## License

This package is open source software.

It is licensed under the Apache License, Version 2.0 (the "License"); you may not use it except in compliance with the License. You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>.


## Getting started

To build, use:

    birch build

To run, use:

    birch sample --config config/linear_regression.json


## Details

The model is given by:

$$\begin{align}
\sigma^2 &\sim \mathrm{Inv\text{-}Gamma}(3, 4/10) \\
\boldsymbol{\beta} &\sim \mathcal{N}(0, I\sigma^2) \\
y_n &\sim \mathcal{N}(\mathbf{x}_n^{\top}\boldsymbol{\beta}, \sigma^2)
\end{align}$$

The parameters are the noise variance $\sigma^2$ and vector of
coefficients $\boldsymbol{\beta}$. The data consists of observations $y_n$
and explanatory variables $\mathbf{x}_n$ for $n=1,\ldots,N$.


## Acknowledgements

This package contains a [bike sharing data set](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) from the Capital Bikeshare system in Washington D.C., studied in [Fanaee-T and Gama (2014)](#references), and prepared in JSON format for Birch.


## References

  1. H. Fanaee-T and J. Gama (2014). [Event labeling combining ensemble detectors and background knowledge](http://dx.doi.org/10.1007/s13748-013-0040-3). *Progress in Artificial Intelligence*. **2**:113-127.
