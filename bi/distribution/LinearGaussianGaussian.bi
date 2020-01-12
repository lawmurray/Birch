/*
 * ed linear-Gaussian-Gaussian random variate.
 */
final class LinearGaussianGaussian(a:Expression<Real>, m:Gaussian,
    c:Expression<Real>, s2:Expression<Real>) < Gaussian(a*m.μ + c,
    a*a*m.σ2 + s2) {
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
   * Likelihood variance.
   */
  auto s2 <- s2;

  function update(x:Real) {
    (m.μ, m.σ2) <- update_linear_gaussian_gaussian(x, a, m.μ, m.σ2, c, s2);
  }

  function downdate(x:Real) {
    (m.μ, m.σ2) <- downdate_linear_gaussian_gaussian(x, a, m.μ, m.σ2, c, s2);
  }
}

function LinearGaussianGaussian(a:Expression<Real>, μ:Gaussian,
    c:Expression<Real>, σ2:Expression<Real>) -> LinearGaussianGaussian {
  m:LinearGaussianGaussian(a, μ, c, σ2);
  μ.setChild(m);
  return m;
}
