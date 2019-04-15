/*
 * Delayed linear-normal-inverse-gamma-log-Gaussian random variate.
 */
final class DelayLinearNormalInverseGammaLogGaussian(x:Random<Real>&, a:Real,
    μ:DelayNormalInverseGamma, c:Real) < DelayValue<Real>(x) {
  /**
   * Scale.
   */
  a:Real <- a;
    
  /**
   * Mean.
   */
  μ:DelayNormalInverseGamma& <- μ;

  /**
   * Offset.
   */
  c:Real <- c;

  function simulate() -> Real {
    return exp(simulate_linear_normal_inverse_gamma_gaussian(a, μ!.μ, c,
        1.0/μ!.λ, μ!.σ2!.α, μ!.σ2!.β));
  }
  
  function observe(x:Real) -> Real {
    return observe_linear_normal_inverse_gamma_gaussian(log(x), a, μ!.μ, c, 
        1.0/μ!.λ, μ!.σ2!.α, μ!.σ2!.β) - log(x);
  }

  function update(x:Real) {
    (μ!.μ, μ!.λ, μ!.σ2!.α, μ!.σ2!.β) <- update_linear_normal_inverse_gamma_gaussian(
        log(x), a, μ!.μ, c, μ!.λ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function downdate(x:Real) {
    (μ!.μ, μ!.λ, μ!.σ2!.α, μ!.σ2!.β) <- downdate_linear_normal_inverse_gamma_gaussian(
        log(x), a, μ!.μ, c, μ!.λ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function pdf(x:Real) -> Real {
    return pdf_linear_normal_inverse_gamma_gaussian(log(x), a, μ!.μ, c, 
        1.0/μ!.λ, μ!.σ2!.α, μ!.σ2!.β)/x;
  }

  function cdf(x:Real) -> Real {
    return cdf_linear_normal_inverse_gamma_gaussian(log(x), a, μ!.μ, c,
        1.0/μ!.λ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function lower() -> Real? {
    return 0.0;
  }
}

function DelayLinearNormalInverseGammaLogGaussian(x:Random<Real>&, a:Real,
    μ:DelayNormalInverseGamma, c:Real) ->
    DelayLinearNormalInverseGammaLogGaussian {
  m:DelayLinearNormalInverseGammaLogGaussian(x, a, μ, c);
  μ.setChild(m);
  return m;
}
