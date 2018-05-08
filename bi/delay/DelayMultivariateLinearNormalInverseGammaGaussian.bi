/*
 * Delayed multivariate linear-normal-inverse-gamma-Gaussian random variate.
 */
class DelayMultivariateLinearNormalInverseGammaGaussian(A:Real[_,_],
    μ:DelayMultivariateNormalInverseGamma, c:Real[_]) <
    DelayValue<Real[_]> {
  /**
   * Scale.
   */
  A:Real[_,_] <- A;

  /**
   * Mean.
   */
  μ:DelayMultivariateNormalInverseGamma <- μ;

  /**
   * Offset.
   */
  c:Real[_] <- c;

  function simulate() -> Real[_] {
    return simulate_multivariate_linear_normal_inverse_gamma_gaussian(A, μ.μ,
        c, μ.Λ, μ.σ2.α, μ.σ2.β);
  }
  
  function observe(x:Real[_]) -> Real {
    return observe_multivariate_linear_normal_inverse_gamma_gaussian(x, A,
        μ.μ, c, μ.Λ, μ.σ2.α, μ.σ2.β);
  }

  function condition(x:Real[_]) {
    (μ.μ, μ.Λ, μ.σ2.α, μ.σ2.β) <- update_multivariate_linear_normal_inverse_gamma_gaussian(
        x, A, μ.μ, c, μ.Λ, μ.σ2.α, μ.σ2.β);
  }

  function pdf(x:Real[_]) -> Real {
    return pdf_multivariate_linear_normal_inverse_gamma_gaussian(
        x, A, μ.μ, c, μ.Λ, μ.σ2.α, μ.σ2.β);
  }
}

function DelayMultivariateLinearNormalInverseGammaGaussian(A:Real[_,_],
    μ:DelayMultivariateNormalInverseGamma, c:Real[_]) ->
    DelayMultivariateLinearNormalInverseGammaGaussian {
  m:DelayMultivariateLinearNormalInverseGammaGaussian(A, μ, c);
  return m;
}
