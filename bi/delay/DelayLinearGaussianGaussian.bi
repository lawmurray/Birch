/*
 * Delayed linear-Gaussian-Gaussian random variate.
 */
class DelayLinearGaussianGaussian(x:Random<Real>&, a:Real, m:DelayGaussian,
    c:Real, s2:Real) < DelayGaussian(x, a*m.μ + c, a*a/m.λ + s2) {
  /**
   * Scale.
   */
  a:Real <- a;
    
  /**
   * Mean.
   */
  m:DelayGaussian& <- m;

  /**
   * Offset.
   */
  c:Real <- c;

  /**
   * Likelihood precision.
   */
  l:Real <- 1.0/s2;

  function update(x:Real) {
    (m!.μ, m!.λ) <- update_linear_gaussian_gaussian(x, a, m!.μ, m!.λ, c, l);
  }
}

function DelayLinearGaussianGaussian(x:Random<Real>&, a:Real,
    μ:DelayGaussian, c:Real, σ2:Real) -> DelayLinearGaussianGaussian {
  m:DelayLinearGaussianGaussian(x, a, μ, c, σ2);
  μ.setChild(m);
  return m;
}
