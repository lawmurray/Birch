/*
 * ed matrix normal-inverse-wishart variate.
 */
final class MatrixNormalInverseWishart(M:Expression<Real[_,_]>,
    U:Expression<Real[_,_]>, V:InverseWishart) < Distribution<Real[_,_]> {
  /**
   * Precision.
   */
  Λ:LLT <- llt(inv(llt(U)));

  /**
   * Precision times mean.
   */
  N:Expression<Real[_,_]> <- Λ*M;
  
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
    return simulate_matrix_normal_inverse_wishart(N.value(), Λ, V.Ψ.value(), V.k.value());
  }
  
  function logpdf(X:Real[_,_]) -> Real {   
    return logpdf_matrix_normal_inverse_wishart(X, N.value(), Λ, V.Ψ.value(), V.k.value());
  }

  function update(X:Real[_,_]) {
    (V.Ψ, V.k) <- update_matrix_normal_inverse_wishart(X, N.value(), Λ, V.Ψ.value(), V.k.value());
  }

  function downdate(X:Real[_,_]) {
    (V.Ψ, V.k) <- downdate_matrix_normal_inverse_wishart(X, N.value(), Λ, V.Ψ.value(), V.k.value());
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "MatrixNormalInverseWishart");
    buffer.set("M", solve(Λ, N.value()));
    buffer.set("Σ", inv(Λ));
    buffer.set("Ψ", V.Ψ.value());
    buffer.set("k", V.k.value());
  }
}

function MatrixNormalInverseWishart(M:Expression<Real[_,_]>,
    U:Expression<Real[_,_]>, V:InverseWishart) -> MatrixNormalInverseWishart {
  m:MatrixNormalInverseWishart(M, U, V);
  V.setChild(m);
  return m;
}
