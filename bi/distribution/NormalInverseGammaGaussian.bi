/*
 * Grafted normal-inverse-gamma-Gaussian distribution.
 */
final class NormalInverseGammaGaussian(μ:NormalInverseGamma) <
    Distribution<Real> {
  /**
   * Mean.
   */
  μ:NormalInverseGamma& <- μ;

  function simulate() -> Real {
    return simulate_normal_inverse_gamma_gaussian(μ.μ.value(),
        1.0/μ.λ.value(), μ.σ2.α.value(), μ.σ2.β.value());
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_normal_inverse_gamma_gaussian(x, μ.μ.value(),
        1.0/μ.λ.value(), μ.σ2.α.value(), μ.σ2.β.value());
  }

  function update(x:Real) {
    (μ.μ, μ.λ, μ.σ2.α, μ.σ2.β) <- update_normal_inverse_gamma_gaussian(
        x, μ.μ.value(), μ.λ.value(), μ.σ2.α.value(), μ.σ2.β.value());
  }

  function downdate(x:Real) {
    (μ.μ, μ.λ, μ.σ2.α, μ.σ2.β) <- downdate_normal_inverse_gamma_gaussian(
        x, μ.μ.value(), μ.λ.value(), μ.σ2.α.value(), μ.σ2.β.value());
  }

  function cdf(x:Real) -> Real? {
    return cdf_normal_inverse_gamma_gaussian(x, μ.μ.value(),
        1.0/μ.λ.value(), μ.σ2.α.value(), μ.σ2.β.value());
  }

  function quantile(P:Real) -> Real? {
    return quantile_normal_inverse_gamma_gaussian(P, μ.μ.value(),
        1.0/μ.λ.value(), μ.σ2.α.value(), μ.σ2.β.value());
  }
}

function NormalInverseGammaGaussian(μ:NormalInverseGamma) ->
    NormalInverseGammaGaussian {
  m:NormalInverseGammaGaussian(μ);
  μ.setChild(m);
  return m;
}
