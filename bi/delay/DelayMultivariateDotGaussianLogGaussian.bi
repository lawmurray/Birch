/*
 * Delayed dot-Gaussian-log-Gaussian random variate. This is univariate, where
 * the prior over the mean is given by a dot product with a multivariate
 * Gaussian random variable, plus scalar.
 */
class DelayMultivariateDotGaussianLogGaussian(x:Random<Real>&, a:Real[_],
    m:DelayMultivariateGaussian, c:Real, s2:Real) <
    DelayLogGaussian(x, dot(a, m.μ) + c, scalar(trans(a)*m.Σ*a) + s2) {
  /**
   * Scale.
   */
  a:Real[_] <- a;
    
  /**
   * Prior mean.
   */
  m:DelayMultivariateGaussian& <- m;

  /**
   * Offset.
   */
  c:Real <- c;

  /**
   * Likelihood variance.
   */
  s2:Real <- s2;

  function condition(x:Real) {
    (m!.μ, m!.Σ) <- update_multivariate_dot_gaussian_gaussian(log(x), a,
        m!.μ, m!.Σ, c, s2);
  }

  function lower() -> Real? {
    return 0.0;
  }
}

function DelayMultivariateDotGaussianLogGaussian(x:Random<Real>&,
    a:Real[_], μ:DelayMultivariateGaussian, c:Real, σ2:Real) ->
    DelayMultivariateDotGaussianLogGaussian {
  m:DelayMultivariateDotGaussianLogGaussian(x, a, μ, c, σ2);
  μ.setChild(m);
  return m;
}
