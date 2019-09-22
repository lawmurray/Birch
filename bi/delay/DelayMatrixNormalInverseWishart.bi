/*
 * Delayed matrix normal-inverse-wishart variate.
 */
final class DelayMatrixNormalInverseWishart(future:Real[_,_]?,
    futureUpdate:Boolean, M:Real[_,_], U:Real[_,_],
    V:DelayInverseWishart) < DelayValue<Real[_,_]>(future,
    futureUpdate) {
  /**
   * Mean.
   */
  M:Real[_,_] <- M;
  
  /**
   * Among-row covariance.
   */
  U:Real[_,_] <- U;
  
  /**
   * Among-column covariance.
   */
  V:DelayInverseWishart& <- V;

  function rows() -> Integer {
    return global.rows(M);
  }
  
  function columns() -> Integer {
    return global.columns(M);
  }

  function simulate() -> Real[_,_] {
    //return simulate_matrix_normal_inverse_wishart(N, Λ, α, γ);
  }
  
  function logpdf(X:Real[_,_]) -> Real {   
    //return logpdf_matrix_normal_inverse_wishart(X, N, Λ, α, γ);
  }

  function update(X:Real[_,_]) {
    //(σ2!.α, σ2!.β) <- update_matrix_normal_inverse_wishart(X, N, Λ, α, γ);
  }

  function downdate(X:Real[_,_]) {
    //(σ2!.α, σ2!.β) <- downdate_matrix_normal_inverse_wishart(X, N, Λ, α, γ);
  }

  function pdf(X:Real[_,_]) -> Real {
    //return pdf_matrix_normal_inverse_wishart(X, N, Λ, α, γ);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "MatrixNormalInverseWishart");
    buffer.set("M", M);
    buffer.set("U", U);
    buffer.set("k", V!.k);
    buffer.set("Ψ", V!.Ψ);
  }
}

function DelayMatrixNormalInverseWishart(future:Real[_,_]?,
    futureUpdate:Boolean, M:Real[_,_], U:Real[_,_], V:DelayInverseWishart) ->
    DelayMatrixNormalInverseWishart {
  m:DelayMatrixNormalInverseWishart(future, futureUpdate, M, U, V);
  V.setChild(m);
  return m;
}
