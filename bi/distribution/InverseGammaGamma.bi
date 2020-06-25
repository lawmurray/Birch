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
    auto θ <- this.θ;
    return simulate_inverse_gamma_gamma(k.value(), θ.α.value(), θ.β.value());
  }

  function simulateLazy() -> Real? {
    auto θ <- this.θ;
    return simulate_inverse_gamma_gamma(k.get(), θ.α.get(), θ.β.get());
  }

  function logpdf(x:Real) -> Real {
    auto θ <- this.θ;
    return logpdf_inverse_gamma_gamma(x, k.value(), θ.α.value(), θ.β.value());
  }

  function logpdfLazy(x:Expression<Real>) -> Expression<Real>? {
    auto θ <- this.θ;
    return logpdf_lazy_inverse_gamma_gamma(x, k, θ.α, θ.β);
  }

  function update(x:Real) {
    auto θ <- this.θ;
    (θ.α, θ.β) <- box(update_inverse_gamma_gamma(x, k.value(), θ.α.value(), θ.β.value()));
  }

  function updateLazy(x:Expression<Real>) {
    auto θ <- this.θ;
    (θ.α, θ.β) <- update_lazy_inverse_gamma_gamma(x, k, θ.α, θ.β);
  }

  function downdate(x:Real) {
    auto θ <- this.θ;
    (θ.α, θ.β) <- box(downdate_inverse_gamma_gamma(x, k.value(), θ.α.value(), θ.β.value()));
  }

  function cdf(x:Real) -> Real? {
    auto θ <- this.θ;
    return cdf_inverse_gamma_gamma(x, k.value(), θ.α.value(), θ.β.value());
  }

  function lower() -> Real? {
    return 0.0;
  }

  function link() {
    auto θ <- this.θ;
    θ.setChild(this);
  }
  
  function unlink() {
    auto θ <- this.θ;
    θ.releaseChild(this);
  }
}

function InverseGammaGamma(k:Expression<Real>, θ:InverseGamma) ->
    InverseGammaGamma {
  m:InverseGammaGamma(k, θ);
  m.link();
  return m;
}
