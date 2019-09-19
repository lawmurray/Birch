/*
 * Delayed dot-normal-inverse-gamma-Gaussian random variate. This is
 * univariate, where the prior over the mean is given by a dot product with a
 * multivariate normal inverse gamma random variable, plus scalar.
 */
final class DelayDotMultivariateNormalInverseGammaGaussian(future:Real?,
    futureUpdate:Boolean, a:Real[_], μ:DelayMultivariateNormalInverseGamma,
    c:Real) < DelayValue<Real>(future, futureUpdate) {
  /**
   * Scale.
   */
  a:Real[_] <- a;

  /**
   * Mean.
   */
  μ:DelayMultivariateNormalInverseGamma& <- μ;

  /**
   * Offset.
   */
  c:Real <- c;

  function simulate() -> Real {
    return simulate_dot_multivariate_normal_inverse_gamma_gaussian(
        a, μ!.ν, c, μ!.Λ, μ!.σ2!.α, μ!.σ2!.β);
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_dot_multivariate_normal_inverse_gamma_gaussian(
        x, a, μ!.ν, c, μ!.Λ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function update(x:Real) {
    (μ!.ν, μ!.Λ, μ!.γ, μ!.σ2!.α, μ!.σ2!.β) <- update_dot_multivariate_normal_inverse_gamma_gaussian(
        x, a, μ!.ν, c, μ!.Λ, μ!.γ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function downdate(x:Real) {
    (μ!.ν, μ!.Λ, μ!.γ, μ!.σ2!.α, μ!.σ2!.β) <- downdate_dot_multivariate_normal_inverse_gamma_gaussian(
        x, a, μ!.ν, c, μ!.Λ, μ!.γ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function pdf(x:Real) -> Real {
    return pdf_dot_multivariate_normal_inverse_gamma_gaussian(
        x, a, μ!.ν, c, μ!.Λ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function cdf(x:Real) -> Real {
    return cdf_dot_multivariate_normal_inverse_gamma_gaussian(
        x, a, μ!.ν, c, μ!.Λ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function write(buffer:Buffer) {
    buffer.set(value());
  }
}

function DelayDotMultivariateNormalInverseGammaGaussian(future:Real?,
    futureUpdate:Boolean, a:Real[_], μ:DelayMultivariateNormalInverseGamma,
    c:Real) -> DelayDotMultivariateNormalInverseGammaGaussian {
  m:DelayDotMultivariateNormalInverseGammaGaussian(future, futureUpdate, a, μ, c);
  μ.setChild(m);
  return m;
}
