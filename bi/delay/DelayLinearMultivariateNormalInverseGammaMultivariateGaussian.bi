/*
 * Delayed linear-normal-inverse-gamma-Gaussian random variate where
 * components have independent and identical variance.
 */
final class DelayLinearMultivariateNormalInverseGammaGaussian(future:Real[_]?,
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

  function simulate() -> Real[_] {
    return simulate_linear_identical_normal_inverse_gamma_gaussian(A,
        solve(μ!.Λ, μ!.ν), c, μ!.Λ, μ!.σ2!.α, μ!.σ2!.β);
  }
  
  function logpdf(x:Real[_]) -> Real {
    return logpdf_linear_identical_normal_inverse_gamma_gaussian(x, A,
        solve(μ!.Λ, μ!.ν), c, μ!.Λ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function update(x:Real[_]) {
    (μ!.ν, μ!.Λ, μ!.γ, μ!.σ2!.α, μ!.σ2!.β) <- update_linear_identical_normal_inverse_gamma_gaussian(
        x, A, μ!.ν, c, μ!.Λ, μ!.γ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function downdate(x:Real[_]) {
    (μ!.ν, μ!.Λ, μ!.γ, μ!.σ2!.α, μ!.σ2!.β) <- downdate_linear_identical_normal_inverse_gamma_gaussian(
        x, A, μ!.ν, c, μ!.Λ, μ!.γ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function pdf(x:Real[_]) -> Real {
    return pdf_linear_identical_normal_inverse_gamma_gaussian(x, A,
        solve(μ!.Λ, μ!.ν), c, μ!.Λ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function write(buffer:Buffer) {
    buffer.set(value());
  }
}

function DelayLinearMultivariateNormalInverseGammaGaussian(future:Real[_]?,
    futureUpdate:Boolean, A:Real[_,_], μ:DelayMultivariateNormalInverseGamma,
    c:Real[_]) -> DelayLinearMultivariateNormalInverseGammaGaussian {
  m:DelayLinearMultivariateNormalInverseGammaGaussian(future, futureUpdate, A, μ, c);
  μ.setChild(m);
  return m;
}
