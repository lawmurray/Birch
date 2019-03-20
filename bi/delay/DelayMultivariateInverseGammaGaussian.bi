/*
 * Delayed multivariate normal-inverse-gamma-Gaussian random variate.
 */
class DelayMultivariateInverseGammaGaussian(x:Random<Real[_]>&, μ:Real[_],
    σ2:DelayInverseGamma) < DelayValue<Real[_]>(x) {
  /**
   * Mean.
   */
  μ:Real[_] <- μ;

  /**
   * Variance.
   */
  σ2:DelayInverseGamma& <- σ2;

  function simulate() -> Real[_] {
    return simulate_multivariate_inverse_gamma_gaussian(μ, σ2!.α, σ2!.β);
  }
  
  function observe(x:Real[_]) -> Real {
    return observe_multivariate_inverse_gamma_gaussian(x, μ, σ2!.α, σ2!.β);
  }

  function update(x:Real[_]) {
    (σ2!.α, σ2!.β) <- update_multivariate_inverse_gamma_gaussian(x, μ, σ2!.α,
        σ2!.β);
  }

  function pdf(x:Real[_]) -> Real {
    return pdf_multivariate_inverse_gamma_gaussian(x, μ, σ2!.α, σ2!.β);
  }
}

function DelayMultivariateInverseGammaGaussian(x:Random<Real[_]>&, μ:Real[_],
    σ2:DelayInverseGamma) -> DelayMultivariateInverseGammaGaussian {
  m:DelayMultivariateInverseGammaGaussian(x, μ, σ2);
  σ2.setChild(m);
  return m;
}
