/*
 * Delayed matrix normal-inverse-wishart variate.
 */
final class DelayMatrixNormalInverseWishart(future:Real[_,_]?,
    futureUpdate:Boolean, M:Real[_,_], U:Real[_,_],
    V:DelayInverseWishart) < DelayValue<Real[_,_]>(future,
    futureUpdate) {
  /**
   * Precision.
   */
  Λ:LLT <- llt(inv(llt(U)));

  /**
   * Precision times mean.
   */
  N:Real[_,_] <- Λ*M;
  
  /**
   * Among-column covariance.
   */
  V:DelayInverseWishart& <- V;

  function rows() -> Integer {
    return global.rows(N);
  }
  
  function columns() -> Integer {
    return global.columns(N);
  }

  function simulate() -> Real[_,_] {
    return simulate_matrix_normal_inverse_wishart(N, Λ, V.Ψ, V.k);
  }
  
  function logpdf(X:Real[_,_]) -> Real {   
    return logpdf_matrix_normal_inverse_wishart(X, N, Λ, V.Ψ, V.k);
  }

  function update(X:Real[_,_]) {
    (V.Ψ, V.k) <- update_matrix_normal_inverse_wishart(X, N, Λ, V.Ψ, V.k);
  }

  function downdate(X:Real[_,_]) {
    (V.Ψ, V.k) <- downdate_matrix_normal_inverse_wishart(X, N, Λ, V.Ψ, V.k);
  }

  function pdf(X:Real[_,_]) -> Real {
    return pdf_matrix_normal_inverse_wishart(X, N, Λ, V.Ψ, V.k);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "MatrixNormalInverseWishart");
    buffer.set("N", N);
    buffer.set("Λ", Λ);
    buffer.set("Ψ", V.Ψ);
    buffer.set("k", V.k);
  }
}

function DelayMatrixNormalInverseWishart(future:Real[_,_]?,
    futureUpdate:Boolean, M:Real[_,_], U:Real[_,_], V:DelayInverseWishart) ->
    DelayMatrixNormalInverseWishart {
  m:DelayMatrixNormalInverseWishart(future, futureUpdate, M, U, V);
  V.setChild(m);
  return m;
}
