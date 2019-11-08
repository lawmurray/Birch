/*
 * Delayed matrix normal-inverse-gamma variate.
 */
final class DelayMatrixNormalInverseGamma(future:Real[_,_]?,
    futureUpdate:Boolean, M:Real[_,_], Σ:Real[_,_],
    σ2:DelayIndependentInverseGamma) < DelayValue<Real[_,_]>(future,
    futureUpdate) {
  /**
   * Precision.
   */
  Λ:LLT <- llt(inv(llt(Σ)));

  /**
   * Precision times mean.
   */
  N:Real[_,_] <- Λ*M;

  /**
   * Variance shapes.
   */
  α:Real <- σ2.α;

  /**
   * Variance scale accumulators.
   */
  γ:Real[_] <- σ2.β + 0.5*diagonal(transpose(N)*M);

  /**
   * Variance scales.
   */
  σ2:DelayIndependentInverseGamma& <- σ2;

  function rows() -> Integer {
    return global.rows(N);
  }
  
  function columns() -> Integer {
    return global.columns(N);
  }

  function simulate() -> Real[_,_] {
    return simulate_matrix_normal_inverse_gamma(N, Λ, α,
        gamma_to_beta(γ, N, Λ));
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    return logpdf_matrix_normal_inverse_gamma(X, N, Λ, α,
        gamma_to_beta(γ, N, Λ));
  }

  function update(X:Real[_,_]) {
    (σ2.α, σ2.β) <- update_matrix_normal_inverse_gamma(X, N, Λ, α,
        gamma_to_beta(γ, N, Λ));
  }

  function downdate(X:Real[_,_]) {
    (σ2.α, σ2.β) <- downdate_matrix_normal_inverse_gamma(X, N, Λ, α,
        gamma_to_beta(γ, N, Λ));
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "MatrixNormalInverseGamma");
    buffer.set("M", solve(Λ, N));
    buffer.set("Σ", inv(Λ));
    buffer.set("α", α);
    buffer.set("β", gamma_to_beta(γ, N, Λ));
  }
}

function DelayMatrixNormalInverseGamma(future:Real[_,_]?,
    futureUpdate:Boolean, M:Real[_,_], Σ:Real[_,_],
    σ2:DelayIndependentInverseGamma) -> DelayMatrixNormalInverseGamma {
  m:DelayMatrixNormalInverseGamma(future, futureUpdate, M, Σ, σ2);
  σ2.setChild(m);
  return m;
}


/*
 * Compute the variance scaleσ from the variance scale accumulatorσ and other
 * parameters.
 */
function gamma_to_beta(γ:Real[_], N:Real[_,_], Λ:LLT) -> Real[_] {
  auto A <- solve(cholesky(Λ), N);
  return γ - 0.5*diagonal(transpose(A)*A);
}
