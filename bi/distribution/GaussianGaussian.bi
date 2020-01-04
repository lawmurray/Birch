/*
 * ed Gaussian-Gaussian random variate.
 */
final class GaussianGaussian(future:Real?, futureUpdate:Boolean,
    m:Gaussian, s2:Expression<Real>) < Gaussian(future, futureUpdate, m.μ,
    1.0/m.λ + s2) {
  /**
   * Mean.
   */
  m:Gaussian& <- m;

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

function GaussianGaussian(future:Real?, futureUpdate:Boolean,
    μ:Gaussian, σ2:Expression<Real>) -> GaussianGaussian {
  m:GaussianGaussian(future, futureUpdate, μ, σ2.graft());
  μ.setChild(m);
  return m;
}
