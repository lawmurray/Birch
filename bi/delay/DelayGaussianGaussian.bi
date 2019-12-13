/*
 * Delayed Gaussian-Gaussian random variate.
 */
final class DelayGaussianGaussian(future:Real?, futureUpdate:Boolean,
    m:DelayGaussian, s2:Expression<Real>) < DelayGaussian(future, futureUpdate, m.μ,
    1.0/m.λ + s2) {
  /**
   * Mean.
   */
  m:DelayGaussian& <- m;

  /**
   * Likelihood precision.
   */
  l:Expression<Real> <- 1.0/s2;

  function update(x:Real) {
    (m.μ, m.λ) <- update_gaussian_gaussian(x, m.μ, m.λ, l);
  }

  function downdate(x:Real) {
    (m.μ, m.λ) <- downdate_gaussian_gaussian(x, m.μ, m.λ, l);
  }
}

function DelayGaussianGaussian(future:Real?, futureUpdate:Boolean,
    μ:DelayGaussian, σ2:Expression<Real>) -> DelayGaussianGaussian {
  m:DelayGaussianGaussian(future, futureUpdate, μ, σ2.graft());
  μ.setChild(m);
  return m;
}
