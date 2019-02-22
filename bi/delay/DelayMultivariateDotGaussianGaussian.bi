/*
 * Delayed dot-Gaussian-Gaussian random variate. This is univariate, where
 * the prior over the mean is given by a dot product with a multivariate
 * Gaussian random variable, plus scalar.
 */
class DelayMultivariateDotGaussianGaussian(x:Random<Real>&,
    a:Real[_], μ_0:DelayMultivariateGaussian, c:Real, σ2:Real) <
    DelayGaussian(x, dot(a, μ_0.μ) + c, scalar(trans(a)*μ_0.Σ*a) + σ2) {
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
    (μ_0!.μ, μ_0!.Σ) <- update_multivariate_dot_gaussian_gaussian(x, a, μ_0!.μ, μ_0!.Σ, μ_m, σ2_m);
  }
}

function DelayMultivariateDotGaussianGaussian(x:Random<Real>&,
    a:Real[_], μ_0:DelayMultivariateGaussian, c:Real, σ2:Real) ->
    DelayMultivariateDotGaussianGaussian {
  m:DelayMultivariateDotGaussianGaussian(x, a, μ_0, c, σ2);
  μ_0.setChild(m);
  return m;
}
