/*
 * Delayed normal-inverse-gamma random variate.
 */
final class DelayNormalInverseGamma(future:Real?, futureUpdate:Boolean,
    μ:Real, a2:Real, σ2:DelayInverseGamma) < DelayValue<Real>(future,
    futureUpdate) {
  /**
   * Mean.
   */
  μ:Real <- μ;
  
  /**
   * Precision scale.
   */
  λ:Real <- 1.0/a2;
  
  /**
   * Variance.
   */
  σ2:DelayInverseGamma& <- σ2;

  function simulate() -> Real {
    return simulate_normal_inverse_gamma(μ, 1.0/λ, σ2!.α, σ2!.β);
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_normal_inverse_gamma(x, μ, 1.0/λ, σ2!.α, σ2!.β);
  }

  function update(x:Real) {
    (σ2!.α, σ2!.β) <- update_normal_inverse_gamma(x, μ, λ, σ2!.α, σ2!.β);
  }

  function downdate(x:Real) {
    (σ2!.α, σ2!.β) <- downdate_normal_inverse_gamma(x, μ, λ, σ2!.α, σ2!.β);
  }

  function pdf(x:Real) -> Real {
    return pdf_normal_inverse_gamma(x, μ, 1.0/λ, σ2!.α, σ2!.β);
  }

  function cdf(x:Real) -> Real {
    return cdf_normal_inverse_gamma(x, μ, 1.0/λ, σ2!.α, σ2!.β);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "NormalInverseGamma");
    buffer.set("μ", μ);
    buffer.set("a2", 1.0/λ);
    buffer.set("α", σ2!.α);
    buffer.set("β", σ2!.β);
  }
}

function DelayNormalInverseGamma(future:Real?, futureUpdate:Boolean, μ:Real,
    a2:Real, σ2:DelayInverseGamma) -> DelayNormalInverseGamma {
  m:DelayNormalInverseGamma(future, futureUpdate, μ, a2, σ2);
  σ2.setChild(m);
  return m;
}
