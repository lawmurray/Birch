/*
 * ed Gaussian-Gaussian random variate.
 */
final class GaussianGaussian(future:Real?, futureUpdate:Boolean,
    m:Gaussian, s2:Expression<Real>) < Gaussian(future, futureUpdate, m.μ,
    m.σ2 + s2) {
  /**
   * Mean.
   */
  m:Gaussian& <- m;

  /**
   * Variance.
   */
  s2:Expression<Real> <- s2;

  function update(x:Real) {
    (m.μ, m.σ2) <- update_gaussian_gaussian(x, m.μ, m.σ2, s2);
  }

  function downdate(x:Real) {
    (m.μ, m.σ2) <- downdate_gaussian_gaussian(x, m.μ, m.σ2, s2);
  }
}

function GaussianGaussian(future:Real?, futureUpdate:Boolean,
    μ:Gaussian, σ2:Expression<Real>) -> GaussianGaussian {
  m:GaussianGaussian(future, futureUpdate, μ, σ2.graft());
  μ.setChild(m);
  return m;
}
