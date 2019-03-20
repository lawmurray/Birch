/*
 * Delayed multivariate Gaussian-Gaussian random variate.
 */
class DelayMultivariateGaussianGaussian(x:Random<Real[_]>&,
    m:DelayMultivariateGaussian, S:Real[_,_]) <
    DelayMultivariateGaussian(x, m.μ, cholinv(m.Λ) + S) {
  /**
   * Mean.
   */
  m:DelayMultivariateGaussian& <- m;

  /**
   * Likelihood precision.
   */
  L:Real[_,_] <- cholinv(S);

  function condition(x:Real[_]) {
    (m!.μ, m!.Λ) <- update_multivariate_gaussian_gaussian(x, m!.μ, m!.Λ, L);
  }
}

function DelayMultivariateGaussianGaussian(x:Random<Real[_]>&,
    μ:DelayMultivariateGaussian, Σ:Real[_,_]) ->
    DelayMultivariateGaussianGaussian {
  m:DelayMultivariateGaussianGaussian(x, μ, Σ);
  μ.setChild(m);
  return m;
}
