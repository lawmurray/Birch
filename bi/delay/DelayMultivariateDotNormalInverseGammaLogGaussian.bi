/*
 * Delayed dot-normal-inverse-gamma-log-Gaussian random variate. This is
 * univariate, where the prior over the mean is given by a dot product with a
 * multivariate normal inverse gamma random variable, plus scalar.
 */
class DelayMultivariateDotNormalInverseGammaLogGaussian(x:Random<Real>&,
    a:Real[_], μ:DelayMultivariateNormalInverseGamma, c:Real) <
    DelayValue<Real>(x) {
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
    return exp(simulate_multivariate_dot_normal_inverse_gamma_gaussian(a, μ!.μ,
        c, μ!.Λ, μ!.σ2!.α, μ!.σ2!.β));
  }
  
  function observe(x:Real) -> Real {
    return observe_multivariate_dot_normal_inverse_gamma_gaussian(log(x), a, μ!.μ,
        c, μ!.Λ, μ!.σ2!.α, μ!.σ2!.β) - log(x);
  }

  function update(x:Real) {
    (μ!.μ, μ!.Λ, μ!.σ2!.α, μ!.σ2!.β) <- update_multivariate_dot_normal_inverse_gamma_gaussian(
        log(x), a, μ!.μ, c, μ!.Λ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function pdf(x:Real) -> Real {
    return pdf_multivariate_dot_normal_inverse_gamma_gaussian(log(x), a, μ!.μ, c,
        μ!.Λ, μ!.σ2!.α, μ!.σ2!.β)/x;
  }

  function cdf(x:Real) -> Real {
    return cdf_multivariate_dot_normal_inverse_gamma_gaussian(log(x), a, μ!.μ, c,
        μ!.Λ, μ!.σ2!.α, μ!.σ2!.β);
  }
  
  function lower() -> Real? {
    return 0.0;
  }
 }

function DelayMultivariateDotNormalInverseGammaLogGaussian(
    x:Random<Real>&, a:Real[_], μ:DelayMultivariateNormalInverseGamma,
    c:Real) -> DelayMultivariateDotNormalInverseGammaLogGaussian {
  m:DelayMultivariateDotNormalInverseGammaLogGaussian(x, a, μ, c);
  μ.setChild(m);
  return m;
}
