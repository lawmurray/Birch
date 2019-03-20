/*
 * Delayed Gaussian-Gaussian random variate.
 */
class DelayGaussianGaussian(x:Random<Real>&, m:DelayGaussian, s2:Real) <
    DelayGaussian(x, m.μ, 1.0/m.λ + s2) {
  /**
   * Mean.
   */
  m:DelayGaussian& <- m;

  /**
   * Likelihood precision.
   */
  l:Real <- 1.0/s2;

  function update(x:Real) {
    (m!.μ, m!.λ) <- update_gaussian_gaussian(x, m!.μ, m!.λ, l);
  }

  function downdate(x:Real) {
    (m!.μ, m!.λ) <- downdate_gaussian_gaussian(x, m!.μ, m!.λ, l);
  }
}

function DelayGaussianGaussian(x:Random<Real>&, μ:DelayGaussian,
    σ2:Real) -> DelayGaussianGaussian {
  m:DelayGaussianGaussian(x, μ, σ2);
  μ.setChild(m);
  return m;
}
