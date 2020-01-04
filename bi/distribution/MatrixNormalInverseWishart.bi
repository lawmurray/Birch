/*
 * ed matrix normal-inverse-wishart variate.
 */
final class MatrixNormalInverseWishart(future:Real[_,_]?,
    futureUpdate:Boolean, M:Real[_,_], U:Real[_,_],
    V:InverseWishart) < Distribution<Real[_,_]>(future,
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
  V:InverseWishart& <- V;

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

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "MatrixNormalInverseWishart");
    buffer.set("M", solve(Λ, N));
    buffer.set("Σ", inv(Λ));
    buffer.set("Ψ", V.Ψ);
    buffer.set("k", V.k);
  }
}

function MatrixNormalInverseWishart(future:Real[_,_]?,
    futureUpdate:Boolean, M:Real[_,_], U:Real[_,_], V:InverseWishart) ->
    MatrixNormalInverseWishart {
  m:MatrixNormalInverseWishart(future, futureUpdate, M, U, V);
  V.setChild(m);
  return m;
}
