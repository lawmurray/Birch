/*
 * Delayed multivariate linear-normal-inverse-gamma-Gaussian random variate.
 */
final class DelayMultivariateLinearNormalInverseGammaGaussian(future:Real[_]?,
    futureUpdate:Boolean, A:Real[_,_], μ:DelayMultivariateNormalInverseGamma,
    c:Real[_]) < DelayValue<Real[_]>(future, futureUpdate) {
  /**
   * Scale.
   */
  A:Real[_,_] <- A;

  /**
   * Mean.
   */
  μ:DelayMultivariateNormalInverseGamma& <- μ;

  /**
   * Offset.
   */
  c:Real[_] <- c;

  /**
   * Covariance scale accumulator,
   * $\gamma = \beta + \frac{1}{2} \nu \Lambda^{-1} \nu$.
   */
  γ:Real <- μ.σ2!.β + 0.5*dot(solve(μ.Λ, μ.ν), μ.ν);

  function simulate() -> Real[_] {
    return simulate_multivariate_linear_normal_inverse_gamma_gaussian(A, solve(μ!.Λ, μ!.ν),
        c, μ!.Λ, μ!.σ2!.α, μ!.σ2!.β);
  }
  
  function observe(x:Real[_]) -> Real {
    return observe_multivariate_linear_normal_inverse_gamma_gaussian(x, A,
        solve(μ!.Λ, μ!.ν), c, μ!.Λ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function update(x:Real[_]) {
    (μ!.ν, μ!.Λ, γ, μ!.σ2!.α, μ!.σ2!.β) <- update_multivariate_linear_normal_inverse_gamma_gaussian(
        x, A, μ!.ν, c, μ!.Λ, γ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function downdate(x:Real[_]) {
    (μ!.ν, μ!.Λ, γ, μ!.σ2!.α, μ!.σ2!.β) <- downdate_multivariate_linear_normal_inverse_gamma_gaussian(
        x, A, μ!.ν, c, μ!.Λ, γ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function pdf(x:Real[_]) -> Real {
    return pdf_multivariate_linear_normal_inverse_gamma_gaussian(x, A, solve(μ!.Λ, μ!.ν),
        c, μ!.Λ, μ!.σ2!.α, μ!.σ2!.β);
  }
}

function DelayMultivariateLinearNormalInverseGammaGaussian(future:Real[_]?,
    futureUpdate:Boolean, A:Real[_,_], μ:DelayMultivariateNormalInverseGamma,
    c:Real[_]) -> DelayMultivariateLinearNormalInverseGammaGaussian {
  m:DelayMultivariateLinearNormalInverseGammaGaussian(future, futureUpdate, A, μ, c);
  μ.setChild(m);
  return m;
}
