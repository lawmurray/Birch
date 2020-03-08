/*
 * ed inverse-gamma-gamma random variate.
 */
final class InverseGammaGamma(k:Expression<Real>, θ:InverseGamma) < Distribution<Real> {
  /**
   * Shape.
   */
  k:Expression<Real> <- k;

  /**
   * Scale.
   */
  θ:InverseGamma& <- θ;

  function simulate() -> Real {
    return simulate_inverse_gamma_gamma(k.value(), θ.α.value(), θ.β.value());
  }

  function logpdf(x:Real) -> Real {
    return logpdf_inverse_gamma_gamma(x, k.value(), θ.α.value(), θ.β.value());
  }

  function update(x:Real) {
    (θ.α, θ.β) <- update_inverse_gamma_gamma(x, k.value(), θ.α.value(), θ.β.value());
  }

  function downdate(x:Real) {
    (θ.α, θ.β) <- downdate_inverse_gamma_gamma(x, k.value(), θ.α.value(), θ.β.value());
  }

  function cdf(x:Real) -> Real? {
    return cdf_inverse_gamma_gamma(x, k.value(), θ.α.value(), θ.β.value());
  }

  function lower() -> Real? {
    return 0.0;
  }
}

function InverseGammaGamma(k:Expression<Real>, θ:InverseGamma) -> InverseGammaGamma {
  m:InverseGammaGamma(k, θ);
  θ.setChild(m);
  return m;
}
