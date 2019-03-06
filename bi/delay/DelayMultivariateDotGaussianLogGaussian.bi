/*
 * Delayed dot-Gaussian-log-Gaussian random variate. This is univariate, where
 * the prior over the mean is given by a dot product with a multivariate
 * Gaussian random variable, plus scalar.
 */
class DelayMultivariateDotGaussianLogGaussian(x:Random<Real>&,
    a:Real[_], μ_0:DelayMultivariateGaussian, c:Real, σ2:Real) <
    DelayLogGaussian(x, dot(a, μ_0.μ) + c, scalar(trans(a)*μ_0.Σ*a) + σ2) {
  /**
   * Scale.
   */
  a:Real[_] <- a;
    
  /**
   * Prior mean.
   */
  μ_0:DelayMultivariateGaussian& <- μ_0;

  /**
   * Marginal mean.
   */
  μ_m:Real <- this.μ;

  /**
   * Marginal variance.
   */
  σ2_m:Real <- this.σ2;

  function condition(x:Real) {
    (μ_0!.μ, μ_0!.Σ) <- update_multivariate_dot_gaussian_gaussian(log(x), a, μ_0!.μ, μ_0!.Σ, μ_m, σ2_m);
  }

  function lower() -> Real? {
    return 0.0;
  }
}

function DelayMultivariateDotGaussianLogGaussian(x:Random<Real>&,
    a:Real[_], μ_0:DelayMultivariateGaussian, c:Real, σ2:Real) ->
    DelayMultivariateDotGaussianLogGaussian {
  m:DelayMultivariateDotGaussianLogGaussian(x, a, μ_0, c, σ2);
  μ_0.setChild(m);
  return m;
}
