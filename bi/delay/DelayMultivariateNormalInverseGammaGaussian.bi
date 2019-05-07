/*
 * Delayed multivariate normal-inverse-gamma-Gaussian random variate.
 */
final class DelayMultivariateNormalInverseGammaGaussian(future:Real[_]?,
    futureUpdate:Boolean, μ:DelayMultivariateNormalInverseGamma) <
    DelayValue<Real[_]>(future, futureUpdate) {
  /**
   * Mean.
   */
  μ:DelayMultivariateNormalInverseGamma& <- μ;

  /**
   * Covariance scale accumulator,
   * $\gamma = \beta + \frac{1}{2} \nu \Lambda^{-1} \nu$.
   */
  γ:Real <- μ.σ2!.β + 0.5*dot(solve(μ.Λ, μ.ν), μ.ν);

  function simulate() -> Real[_] {
    return simulate_multivariate_normal_inverse_gamma_gaussian(solve(μ!.Λ, μ!.ν), μ!.Λ,
        μ!.σ2!.α, μ!.σ2!.β);
  }
  
  function observe(x:Real[_]) -> Real {
    return observe_multivariate_normal_inverse_gamma_gaussian(x, solve(μ!.Λ, μ!.ν), μ!.Λ,
        μ!.σ2!.α, μ!.σ2!.β);
  }

  function update(x:Real[_]) {
    (μ!.ν, μ!.Λ, γ, μ!.σ2!.α, μ!.σ2!.β) <- update_multivariate_normal_inverse_gamma_gaussian(
        x, μ!.ν, μ!.Λ, γ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function downdate(x:Real[_]) {
    (μ!.ν, μ!.Λ, γ, μ!.σ2!.α, μ!.σ2!.β) <- downdate_multivariate_normal_inverse_gamma_gaussian(
        x, μ!.ν, μ!.Λ, γ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function pdf(x:Real[_]) -> Real {
    return pdf_multivariate_normal_inverse_gamma_gaussian(x, solve(μ!.Λ, μ!.ν), μ!.Λ,
        μ!.σ2!.α, μ!.σ2!.β);
  }
}

function DelayMultivariateNormalInverseGammaGaussian(future:Real[_]?,
    futureUpdate:Boolean, μ:DelayMultivariateNormalInverseGamma) ->
    DelayMultivariateNormalInverseGammaGaussian {
  m:DelayMultivariateNormalInverseGammaGaussian(future, futureUpdate, μ);
  μ.setChild(m);
  return m;
}
