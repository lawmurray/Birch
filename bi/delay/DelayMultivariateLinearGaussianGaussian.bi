/*
 * Delayed multivariate linear-Gaussian-Gaussian random variate.
 */
class DelayMultivariateLinearGaussianGaussian(x:Random<Real[_]>&,
    A:Real[_,_], μ_0:DelayMultivariateGaussian, c:Real[_], Σ:Real[_,_]) <
    DelayMultivariateGaussian(x, A*μ_0.μ + c, A*μ_0.Σ*trans(A) + Σ) {
  /**
   * Scale.
   */
  A:Real[_,_] <- A;
    
  /**
   * Prior mean.
   */
  μ_0:DelayMultivariateGaussian& <- μ_0;

  /**
   * Offset.
   */
  c:Real[_] <- c;

  /**
   * Marginal mean.
   */
  μ_m:Real[_] <- this.μ;

  /**
   * Marginal variance.
   */
  Σ_m:Real[_,_] <- this.Σ;

  function condition(x:Real[_]) {
    (μ_0!.μ, μ_0!.Σ) <- update_multivariate_linear_gaussian_gaussian(x, A,
        μ_0!.μ, μ_0!.Σ, μ_m, Σ_m);
  }
}

function DelayMultivariateLinearGaussianGaussian(x:Random<Real[_]>&,
    A:Real[_,_], μ_0:DelayMultivariateGaussian, c:Real[_], Σ:Real[_,_]) ->
    DelayMultivariateLinearGaussianGaussian {
  m:DelayMultivariateLinearGaussianGaussian(x, A, μ_0, c, Σ);
  μ_0.setChild(m);
  return m;
}
