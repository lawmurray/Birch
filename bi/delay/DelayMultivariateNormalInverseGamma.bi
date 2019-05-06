/*
 * Delayed multivariate normal-inverse-gamma random variate.
 */
final class DelayMultivariateNormalInverseGamma(future:Real[_]?,
    futureUpdate:Boolean, μ:Real[_], A:Real[_,_], σ2:DelayInverseGamma) <
    DelayValue<Real[_]>(future, futureUpdate) {
  /**
   * Mean.
   */
  μ:Real[_] <- μ;

  /**
   * Precision.
   */
  Λ:LLT <- llt(cholinv(A));

  /**
   * Scale.
   */
  σ2:DelayInverseGamma& <- σ2;

  function size() -> Integer {
    return length(μ);
  }

  function simulate() -> Real[_] {
    return simulate_multivariate_normal_inverse_gamma(μ, Λ, σ2!.α, σ2!.β);
  }
  
  function observe(x:Real[_]) -> Real {
    return observe_multivariate_normal_inverse_gamma(x, μ, Λ, σ2!.α, σ2!.β);
  }

  function update(x:Real[_]) {
    (σ2!.α, σ2!.β) <- update_multivariate_normal_inverse_gamma(x, μ, Λ,
        σ2!.α, σ2!.β);
  }

  function downdate(x:Real[_]) {
    (σ2!.α, σ2!.β) <- downdate_multivariate_normal_inverse_gamma(x, μ, Λ,
        σ2!.α, σ2!.β);
  }
  
  function pdf(x:Real[_]) -> Real {
    return pdf_multivariate_normal_inverse_gamma(x, μ, Λ, σ2!.α, σ2!.β);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "MultivariateNormalInverseGamma");
    buffer.set("μ", μ);
    buffer.set("A", inv(Λ));
    buffer.set("α", σ2!.α);
    buffer.set("β", σ2!.β);
  }
}

function DelayMultivariateNormalInverseGamma(future:Real[_]?,
    futureUpdate:Boolean, μ:Real[_], A:Real[_,_], σ2:DelayInverseGamma) ->
    DelayMultivariateNormalInverseGamma {
  m:DelayMultivariateNormalInverseGamma(future, futureUpdate, μ, A, σ2);
  σ2.setChild(m);
  return m;
}
