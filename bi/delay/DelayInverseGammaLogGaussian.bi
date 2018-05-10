/*
 * Delayed normal-inverse-gamma-log-Gaussian random variate.
 */
class DelayInverseGammaLogGaussian(x:Random<Real>&, μ:Real,
    σ2:DelayInverseGamma) < DelayValue<Real>(x) {
  /**
   * Mean.
   */
  μ:Real <- μ;

  /**
   * Variance.
   */
  σ2:DelayInverseGamma <- σ2;

  function simulate() -> Real {
    return exp(simulate_inverse_gamma_gaussian(μ, σ2.α, σ2.β));
  }
  
  function observe(x:Real) -> Real {
    return observe_inverse_gamma_gaussian(log(x), μ, σ2.α, σ2.β) - log(x);
  }

  function condition(x:Real) {
    (σ2.α, σ2.β) <- update_inverse_gamma_gaussian(log(x), μ, σ2.α, σ2.β);
  }

  function pdf(x:Real) -> Real {
    return pdf_inverse_gamma_gaussian(log(x), μ, σ2.α, σ2.β)/x;
  }

  function cdf(x:Real) -> Real {
    return cdf_inverse_gamma_gaussian(log(x), μ, σ2.α, σ2.β);
  }
}

function DelayInverseGammaLogGaussian(x:Random<Real>&, μ:Real,
    σ2:DelayInverseGamma) -> DelayInverseGammaLogGaussian {
  m:DelayInverseGammaLogGaussian(x, μ, σ2);
  σ2.setChild(m);
  return m;
}
