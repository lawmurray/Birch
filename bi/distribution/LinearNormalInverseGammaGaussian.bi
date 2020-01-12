/*
 * ed linear-normal-inverse-gamma-Gaussian random variate.
 */
final class LinearNormalInverseGammaGaussian(a:Real, μ:NormalInverseGamma,
    c:Real) < Distribution<Real> {
  /**
   * Scale.
   */
  a:Real <- a;
    
  /**
   * Mean.
   */
  μ:NormalInverseGamma& <- μ;

  /**
   * Offset.
   */
  c:Real <- c;

  function simulate() -> Real {
    return simulate_linear_normal_inverse_gamma_gaussian(a, μ.μ, 1.0/μ.λ, c,
        μ.σ2.α, μ.σ2.β);
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_linear_normal_inverse_gamma_gaussian(x, a, μ.μ, 1.0/μ.λ, c,
        μ.σ2.α, μ.σ2.β);
  }

  function update(x:Real) {
    (μ.μ, μ.λ, μ.σ2.α, μ.σ2.β) <- update_linear_normal_inverse_gamma_gaussian(
        x, a, μ.μ, μ.λ, c, μ.σ2.α, μ.σ2.β);
  }

  function downdate(x:Real) {
    (μ.μ, μ.λ, μ.σ2.α, μ.σ2.β) <- downdate_linear_normal_inverse_gamma_gaussian(
        x, a, μ.μ, μ.λ, c, μ.σ2.α, μ.σ2.β);
  }

  function cdf(x:Real) -> Real? {
    return cdf_linear_normal_inverse_gamma_gaussian(x, a, μ.μ, 1.0/μ.λ, c,
        μ.σ2.α, μ.σ2.β);
  }

  function quantile(P:Real) -> Real? {
    return quantile_linear_normal_inverse_gamma_gaussian(P, a, μ.μ, 1.0/μ.λ,
        c, μ.σ2.α, μ.σ2.β);
  }
}

function LinearNormalInverseGammaGaussian(a:Real, μ:NormalInverseGamma,
    c:Real) -> LinearNormalInverseGammaGaussian {
  m:LinearNormalInverseGammaGaussian(a, μ, c);
  μ.setChild(m);
  return m;
}
