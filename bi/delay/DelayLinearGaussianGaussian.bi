/*
 * Delayed linear-Gaussian-Gaussian random variate.
 */
final class DelayLinearGaussianGaussian(future:Real?, futureUpdate:Boolean,
    a:Expression<Real>, m:DelayGaussian, c:Expression<Real>,
    s2:Expression<Real>) < DelayGaussian(future, futureUpdate, a*m.μ + c,
    a*a/m.λ + s2) {
  /**
   * Scale.
   */
  auto a <- a;
    
  /**
   * Mean.
   */
  m:DelayGaussian& <- m;

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

function DelayLinearGaussianGaussian(future:Real?, futureUpdate:Boolean,
    a:Expression<Real>, μ:DelayGaussian, c:Expression<Real>,
    σ2:Expression<Real>) -> DelayLinearGaussianGaussian {
  m:DelayLinearGaussianGaussian(future, futureUpdate, a.graft(), μ,
      c.graft(), σ2.graft());
  μ.setChild(m);
  return m;
}
