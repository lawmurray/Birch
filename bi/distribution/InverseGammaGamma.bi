/**
 * Inverse-gamma-gamma distribution.
 */
final class InverseGammaGamma(k:Expression<Real>, θ:InverseGamma) <
    Distribution<Real> {
  /**
   * Shape.
   */
  k:Expression<Real> <- k;

  /**
   * Scale.
   */
  θ:InverseGamma& <- θ;

  function supportsLazy() -> Boolean {
    return true;
  }

  function simulate() -> Real {
    return simulate_inverse_gamma_gamma(k.value(), θ.α.value(), θ.β.value());
  }

  function simulateLazy() -> Real? {
    return simulate_inverse_gamma_gamma(k.get(), θ.α.get(), θ.β.get());
  }

  function logpdf(x:Real) -> Real {
    return logpdf_inverse_gamma_gamma(x, k.value(), θ.α.value(), θ.β.value());
  }

  function logpdfLazy(x:Expression<Real>) -> Expression<Real>? {
    return logpdf_lazy_inverse_gamma_gamma(x, k, θ.α, θ.β);
  }

  function update(x:Real) {
    (θ.α, θ.β) <- box(update_inverse_gamma_gamma(x, k.value(), θ.α.value(), θ.β.value()));
  }

  function updateLazy(x:Expression<Real>) {
    (θ.α, θ.β) <- update_lazy_inverse_gamma_gamma(x, k, θ.α, θ.β);
  }

  function downdate(x:Real) {
    (θ.α, θ.β) <- box(downdate_inverse_gamma_gamma(x, k.value(), θ.α.value(), θ.β.value()));
  }

  function cdf(x:Real) -> Real? {
    return cdf_inverse_gamma_gamma(x, k.value(), θ.α.value(), θ.β.value());
  }

  function lower() -> Real? {
    return 0.0;
  }

  function link() {
    θ.setChild(this);
  }
  
  function unlink() {
    θ.releaseChild(this);
  }
}

function InverseGammaGamma(k:Expression<Real>, θ:InverseGamma) ->
    InverseGammaGamma {
  m:InverseGammaGamma(k, θ);
  m.link();
  return m;
}
