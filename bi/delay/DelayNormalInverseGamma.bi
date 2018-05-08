/*
 * Delayed normal-inverse-gamma random variate.
 */
class DelayNormalInverseGamma(μ:Real, a2:Real, σ2:DelayInverseGamma) <
    DelayValue<Real> {
  /**
   * Mean.
   */
  μ:Real <- μ;
  
  /**
   * Scale.
   */
  a2:Real <- a2;
  
  /**
   * Variance.
   */
  σ2:DelayInverseGamma <- σ2;

  function simulate() -> Real {
    return simulate_normal_inverse_gamma(μ, a2, σ2.α, σ2.β);
  }
  
  function observe(x:Real) -> Real {
    return observe_normal_inverse_gamma(x, μ, a2, σ2.α, σ2.β);
  }

  function pdf(x:Integer) -> Real {
    return pdf_normal_inverse_gamma(x, μ, a2, σ2.α, σ2.β);
  }

  function cdf(x:Integer) -> Real {
    return cdf_normal_inverse_gamma(x, μ, a2, σ2.α, σ2.β);
  }
}

function DelayNormalInverseGamma(μ:Real, a2:Real, σ2:DelayInverseGamma) ->
    DelayNormalInverseGamma {
  m:DelayNormalInverseGamma(μ, a2, σ2);
  return m;
}
