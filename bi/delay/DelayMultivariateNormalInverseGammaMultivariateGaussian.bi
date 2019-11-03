/*
 * Delayed multivariate normal-inverse-gamma-Gaussian random variate.
 */
final class DelayMultivariateNormalInverseGammaMultivariateGaussian(
    future:Real[_]?, futureUpdate:Boolean,
    μ:DelayMultivariateNormalInverseGamma) < DelayValue<Real[_]>(future, futureUpdate) {
  /**
   * Mean.
   */
  μ:DelayMultivariateNormalInverseGamma& <- μ;

  function simulate() -> Real[_] {
    return simulate_multivariate_normal_inverse_gamma_multivariate_gaussian(
        μ.ν, μ.Λ, μ.α, μ.γ);
  }
  
  function logpdf(x:Real[_]) -> Real {
    return logpdf_multivariate_normal_inverse_gamma_multivariate_gaussian(x,
        μ.ν, μ.Λ, μ.α, μ.γ);
  }

  function update(x:Real[_]) {
    (μ.ν, μ.Λ, μ.α, μ.γ) <- update_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, μ.ν, μ.Λ, μ.α, μ.γ);
  }

  function downdate(x:Real[_]) {
    (μ.ν, μ.Λ, μ.α, μ.γ) <- downdate_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, μ.ν, μ.Λ, μ.α, μ.γ);
  }

  function write(buffer:Buffer) {
    buffer.set(value());
  }
}

function DelayMultivariateNormalInverseGammaMultivariateGaussian(future:Real[_]?,
    futureUpdate:Boolean, μ:DelayMultivariateNormalInverseGamma) ->
    DelayMultivariateNormalInverseGammaMultivariateGaussian {
  m:DelayMultivariateNormalInverseGammaMultivariateGaussian(future, futureUpdate, μ);
  μ.setChild(m);
  return m;
}
