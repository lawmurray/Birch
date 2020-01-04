/*
 * ed linear-Gaussian-Gaussian random variate.
 */
final class LinearGaussianGaussian(future:Real?, futureUpdate:Boolean,
    a:Expression<Real>, m:Gaussian, c:Expression<Real>,
    s2:Expression<Real>) < Gaussian(future, futureUpdate, a*m.μ + c,
    a*a/m.λ + s2) {
  /**
   * Scale.
   */
  auto a <- a;
    
  /**
   * Mean.
   */
  m:Gaussian& <- m;

  /**
   * Offset.
   */
  auto c <- c;

  /**
   * Likelihood precision.
   */
  auto l <- 1.0/s2;

  function update(x:Real) {
    (m.μ, m.λ) <- update_linear_gaussian_gaussian(x, a, m.μ, m.λ, c, l);
  }

  function downdate(x:Real) {
    (m.μ, m.λ) <- downdate_linear_gaussian_gaussian(x, a, m.μ, m.λ, c, l);
  }
}

function LinearGaussianGaussian(future:Real?, futureUpdate:Boolean,
    a:Expression<Real>, μ:Gaussian, c:Expression<Real>,
    σ2:Expression<Real>) -> LinearGaussianGaussian {
  m:LinearGaussianGaussian(future, futureUpdate, a.graft(), μ,
      c.graft(), σ2.graft());
  μ.setChild(m);
  return m;
}
