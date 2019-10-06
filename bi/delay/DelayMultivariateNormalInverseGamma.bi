/*
 * Delayed multivariate normal-inverse-gamma random variate.
 */
final class DelayMultivariateNormalInverseGamma(future:Real[_]?,
    futureUpdate:Boolean, μ:Real[_], Σ:Real[_,_], σ2:DelayInverseGamma) <
    DelayValue<Real[_]>(future, futureUpdate) {
  /**
   * Precision.
   */
  Λ:LLT <- llt(inv(llt(Σ)));

  /**
   * Precision times mean.
   */
  ν:Real[_] <- Λ*μ;

  /**
   * Variance shape.
   */
  α:Real <- σ2.α;

  /**
   * Variance scale accumulator.
   */
  γ:Real <- σ2.β + 0.5*dot(μ, ν);

  /**
   * Variance scale.
   */
  σ2:DelayInverseGamma& <- σ2;

  function size() -> Integer {
    return length(ν);
  }

  function simulate() -> Real[_] {
    return simulate_multivariate_normal_inverse_gamma(ν, Λ, α, gamma_to_beta(γ, ν, Λ));
  }
  
  function logpdf(x:Real[_]) -> Real {
    return logpdf_multivariate_normal_inverse_gamma(x, ν, Λ, α, gamma_to_beta(γ, ν, Λ));
  }

  function update(x:Real[_]) {
    (σ2.α, σ2.β) <- update_multivariate_normal_inverse_gamma(x, ν, Λ, α, gamma_to_beta(γ, ν, Λ));
  }

  function downdate(x:Real[_]) {
    (σ2.α, σ2.β) <- downdate_multivariate_normal_inverse_gamma(x, ν, Λ, α, gamma_to_beta(γ, ν, Λ));
  }
  
  function pdf(x:Real[_]) -> Real {
    return pdf_multivariate_normal_inverse_gamma(x, ν, Λ, α, gamma_to_beta(γ, ν, Λ));
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "MultivariateNormalInverseGamma");
    buffer.set("ν", ν);
    buffer.set("Λ", matrix(Λ));
    buffer.set("α", α);
    buffer.set("γ", γ);
  }
}

function DelayMultivariateNormalInverseGamma(future:Real[_]?,
    futureUpdate:Boolean, μ:Real[_], Σ:Real[_,_], σ2:DelayInverseGamma) ->
    DelayMultivariateNormalInverseGamma {
  m:DelayMultivariateNormalInverseGamma(future, futureUpdate, μ, Σ, σ2);
  σ2.setChild(m);
  return m;
}

/*
 * Compute the variance scale from the variance scale accumulator and other
 * parameters.
 */
function gamma_to_beta(γ:Real, ν:Real[_], Λ:LLT) -> Real {
  return γ - 0.5*dot(solve(cholesky(Λ), ν));
}
