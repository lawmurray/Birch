/*
 * Delayed normal-inverse-gamma-log-Gaussian random variate.
 */
final class DelayNormalInverseGammaLogGaussian(x:Random<Real>&,
    μ:DelayNormalInverseGamma) < DelayValue<Real>(x) {
  /**
   * Mean.
   */
  μ:DelayNormalInverseGamma& <- μ;

  function simulate() -> Real {
    return exp(simulate_normal_inverse_gamma_gaussian(μ!.μ, 1.0/μ!.λ,
        μ!.σ2!.α, μ!.σ2!.β));
  }
  
  function observe(x:Real) -> Real {
    return observe_normal_inverse_gamma_gaussian(log(x), μ!.μ, 1.0/μ!.λ,
        μ!.σ2!.α, μ!.σ2!.β) - log(x);
  }

  function update(x:Real) {
    (μ!.μ, μ!.λ, μ!.σ2!.α, μ!.σ2!.β) <- update_normal_inverse_gamma_gaussian(
        log(x), μ!.μ, μ!.λ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function downdate(x:Real) {
    (μ!.μ, μ!.λ, μ!.σ2!.α, μ!.σ2!.β) <- downdate_normal_inverse_gamma_gaussian(
        log(x), μ!.μ, μ!.λ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function pdf(x:Integer) -> Real {
    return pdf_normal_inverse_gamma_gaussian(log(x), μ!.μ, 1.0/μ!.λ,
        μ!.σ2!.α, μ!.σ2!.β)/x;
  }

  function cdf(x:Integer) -> Real {
    return cdf_normal_inverse_gamma_gaussian(log(x), μ!.μ, 1.0/μ!.λ,
        μ!.σ2!.α, μ!.σ2!.β);
  }

  function lower() -> Real? {
    return 0.0;
  }
}

function DelayNormalInverseGammaLogGaussian(x:Random<Real>&,
    μ:DelayNormalInverseGamma) -> DelayNormalInverseGammaLogGaussian {
  m:DelayNormalInverseGammaLogGaussian(x, μ);
  μ.setChild(m);
  return m;
}
