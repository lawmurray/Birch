/*
 * Delayed dot-normal-inverse-gamma-log-Gaussian random variate. This is
 * univariate, where the prior over the mean is given by a dot product with a
 * multivariate normal inverse gamma random variable, plus scalar.
 */
final class DelayDotMultivariateNormalInverseGammaLogGaussian(future:Real?,
    futureUpdate:Boolean, a:Real[_], μ:DelayIdenticalNormalInverseGamma,
    c:Real) < DelayValue<Real>(future, futureUpdate) {
  /**
   * Scale.
   */
  a:Real[_] <- a;

  /**
   * Mean.
   */
  μ:DelayIdenticalNormalInverseGamma& <- μ;

  /**
   * Offset.
   */
  c:Real <- c;

  function simulate() -> Real {
    return exp(simulate_dot_identical_normal_inverse_gamma_gaussian(a, solve(μ!.Λ, μ!.ν),
        c, μ!.Λ, μ!.σ2!.α, μ!.σ2!.β));
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_dot_identical_normal_inverse_gamma_gaussian(log(x), a, solve(μ!.Λ, μ!.ν),
        c, μ!.Λ, μ!.σ2!.α, μ!.σ2!.β) - log(x);
  }

  function update(x:Real) {
    (μ!.ν, μ!.Λ, μ!.γ, μ!.σ2!.α, μ!.σ2!.β) <- update_dot_identical_normal_inverse_gamma_gaussian(
        log(x), a, μ!.ν, c, μ!.Λ, μ!.γ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function downdate(x:Real) {
    (μ!.ν, μ!.Λ, μ!.γ, μ!.σ2!.α, μ!.σ2!.β) <- downdate_dot_identical_normal_inverse_gamma_gaussian(
        log(x), a, μ!.ν, c, μ!.Λ, μ!.γ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function pdf(x:Real) -> Real {
    return pdf_dot_identical_normal_inverse_gamma_gaussian(log(x), a, solve(μ!.Λ, μ!.ν), c,
        μ!.Λ, μ!.σ2!.α, μ!.σ2!.β)/x;
  }

  function cdf(x:Real) -> Real {
    return cdf_dot_identical_normal_inverse_gamma_gaussian(log(x), a, solve(μ!.Λ, μ!.ν), c,
        μ!.Λ, μ!.σ2!.α, μ!.σ2!.β);
  }
  
  function lower() -> Real? {
    return 0.0;
  }

  function write(buffer:Buffer) {
    buffer.set(value());
  }
 }

function DelayDotMultivariateNormalInverseGammaLogGaussian(future:Real?,
    futureUpdate:Boolean, a:Real[_], μ:DelayIdenticalNormalInverseGamma,
    c:Real) -> DelayDotMultivariateNormalInverseGammaLogGaussian {
  m:DelayDotMultivariateNormalInverseGammaLogGaussian(future, futureUpdate, a, μ, c);
  μ.setChild(m);
  return m;
}
