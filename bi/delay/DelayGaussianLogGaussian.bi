/*
 * Delayed Gaussian-log-Gaussian random variate.
 */
class DelayGaussianLogGaussian(x:Random<Real>&, m:DelayGaussian, s2:Real) <
    DelayLogGaussian(x, m.μ, 1.0/m.λ + s2) {
  /**
   * Mean.
   */
  m:DelayGaussian& <- m;

  /**
   * Likelihood precision.
   */
  l:Real <- 1.0/s2;

  function condition(x:Real) {
    (m!.μ, m!.λ) <- update_gaussian_gaussian(log(x), m!.μ, m!.λ, l);
  }
}

function DelayGaussianLogGaussian(x:Random<Real>&, μ:DelayGaussian,
    σ2:Real) -> DelayGaussianLogGaussian {
  m:DelayGaussianLogGaussian(x, μ, σ2);
  μ.setChild(m);
  return m;
}
