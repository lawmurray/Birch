/**
 * Linear-Gaussian-Gaussian distribution.
 */
final class LinearGaussianGaussian(a:Expression<Real>, m:Gaussian,
    c:Expression<Real>, s2:Expression<Real>) < Gaussian(a*m.μ + c,
    a*a*m.σ2 + s2) {
  /**
   * Scale.
   */
  a:Expression<Real> <- a;
    
  /**
   * Mean.
   */
  m:Gaussian <- m;

  /**
   * Offset.
   */
  c:Expression<Real> <- c;

  /**
   * Likelihood variance.
   */
  s2:Expression<Real> <- s2;

  function update(x:Real) {
    (m.μ, m.σ2) <- box(update_linear_gaussian_gaussian(x, a.value(), m.μ.value(), m.σ2.value(), c.value(), s2.value()));
  }

  function downdate(x:Real) {
    (m.μ, m.σ2) <- box(downdate_linear_gaussian_gaussian(x, a.value(), m.μ.value(), m.σ2.value(), c.value(), s2.value()));
  }

  function updateLazy(x:Expression<Real>) {
    (m.μ, m.σ2) <- update_lazy_linear_gaussian_gaussian(x, a, m.μ, m.σ2, c, s2);
  }

  function link() {
    m.setChild(this);
  }
  
  function unlink() {
    m.releaseChild(this);
  }
}

function LinearGaussianGaussian(a:Expression<Real>, μ:Gaussian,
    c:Expression<Real>, σ2:Expression<Real>) -> LinearGaussianGaussian {
  m:LinearGaussianGaussian(a, μ, c, σ2);
  m.link();
  return m;
}
