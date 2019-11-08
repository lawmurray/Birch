/*
 * Delayed linear-normal-inverse-gamma-Gaussian random variate.
 */
final class DelayLinearNormalInverseGammaGaussian(future:Real?,
    futureUpdate:Boolean, a:Real, μ:DelayNormalInverseGamma, c:Real) <
    DelayValue<Real>(future, futureUpdate) {
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

  function quantile(p:Real) -> Real? {
    return quantile_linear_normal_inverse_gamma_gaussian(p, a, μ.μ, 1.0/μ.λ,
        c, μ.σ2.α, μ.σ2.β);
  }

  function write(buffer:Buffer) {
    buffer.set(value());
  }
}

function DelayLinearNormalInverseGammaGaussian(future:Real?,
    futureUpdate:Boolean, a:Real, μ:DelayNormalInverseGamma, c:Real) ->
    DelayLinearNormalInverseGammaGaussian {
  m:DelayLinearNormalInverseGammaGaussian(future, futureUpdate, a, μ, c);
  μ.setChild(m);
  return m;
}
