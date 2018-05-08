/*
 * Delayed linear-normal-inverse-gamma-Gaussian random variate.
 */
class DelayLinearNormalInverseGammaGaussian(a:Real, μ:DelayNormalInverseGamma,
    c:Real) < DelayValue<Real> {
  /**
   * Scale.
   */
  a:Real <- a;
    
  /**
   * Mean.
   */
  μ:DelayNormalInverseGamma <- μ;

  /**
   * Offset.
   */
  c:Real <- c;

  function simulate() -> Real {
    return simulate_linear_normal_inverse_gamma_gaussian(a, μ.μ, c, μ.a2,
        μ.σ2.α, μ.σ2.β);
  }
  
  function observe(x:Real) -> Real {
    return observe_linear_normal_inverse_gamma_gaussian(x, a, μ.μ, c,
        μ.a2, μ.σ2.α, μ.σ2.β);
  }

  function condition(x:Real) {
    (μ.μ, μ.a2, μ.σ2.α, μ.σ2.β) <- update_linear_normal_inverse_gamma_gaussian(
        x, a, μ.μ, c, μ.a2, μ.σ2.α, μ.σ2.β);
  }

  function pdf(x:Real) -> Real {
    return pdf_linear_normal_inverse_gamma_gaussian(x, a, μ.μ, c, μ.a2,
        μ.σ2.α, μ.σ2.β);
  }

  function cdf(x:Real) -> Real {
    return cdf_linear_normal_inverse_gamma_gaussian(x, a, μ.μ, c, μ.a2,
        μ.σ2.α, μ.σ2.β);
  }
}

function DelayLinearNormalInverseGammaGaussian(a:Real,
    μ:DelayNormalInverseGamma, c:Real) ->
    DelayLinearNormalInverseGammaGaussian {
  m:DelayLinearNormalInverseGammaGaussian(a, μ, c);
  return m;
}
