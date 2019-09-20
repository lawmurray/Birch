/*
 * Delayed multivariate Gaussian variate with dot matrix normal inverse
 * gamma prior.
 */
final class DelayDotMatrixNormalInverseGammaMultivariateGaussian(
    future:Real[_]?, futureUpdate:Boolean, a:Real[_],
    θ:DelayMatrixNormalInverseGamma) < DelayValue<Real[_]>(future,
    futureUpdate) {
  /**
   * Scale.
   */
  a:Real[_] <- a;

  /**
   * Parameters.
   */
  θ:DelayMatrixNormalInverseGamma& <- θ;

  function simulate() -> Real[_] {
    return simulate_dot_matrix_normal_inverse_gamma_multivariate_gaussian(a, θ!.N, θ!.Λ, θ!.α, θ!.γ);
  }
  
  function logpdf(x:Real[_]) -> Real {
    return logpdf_dot_matrix_normal_inverse_gamma_multivariate_gaussian(x, a, θ!.N, θ!.Λ, θ!.α, θ!.γ);
  }

  function update(x:Real[_]) {
    (θ!.N, θ!.Λ, θ!.α, θ!.γ) <- update_dot_matrix_normal_inverse_gamma_multivariate_gaussian(x, a, θ!.N, θ!.Λ, θ!.α, θ!.γ);
  }

  function downdate(x:Real[_]) {
    (θ!.N, θ!.Λ, θ!.α, θ!.γ) <- downdate_dot_matrix_normal_inverse_gamma_multivariate_gaussian(x, a, θ!.N, θ!.Λ, θ!.α, θ!.γ);
  }

  function pdf(x:Real[_]) -> Real {
    return pdf_dot_matrix_normal_inverse_gamma_multivariate_gaussian(x, a, θ!.N, θ!.Λ, θ!.α, θ!.γ);
  }

  function write(buffer:Buffer) {
    buffer.set(value());
  }
}

function DelayDotMatrixNormalInverseGammaMultivariateGaussian(
    future:Real[_]?, futureUpdate:Boolean, a:Real[_],
    θ:DelayMatrixNormalInverseGamma) ->
    DelayDotMatrixNormalInverseGammaMultivariateGaussian {
  m:DelayDotMatrixNormalInverseGammaMultivariateGaussian(future,
      futureUpdate, a, θ);
  return m;
}
