/*
 * Grafted matrix normal-inverse-Wishart distribution.
 */
final class MatrixNormalInverseWishart(M:Expression<Real[_,_]>,
    U:Expression<Real[_,_]>, V:InverseWishart) < Distribution<Real[_,_]> {
  /**
   * Precision.
   */
  Λ:Expression<LLT> <- llt(inv(llt(U)));

  /**
   * Precision times mean.
   */
  N:Expression<Real[_,_]> <- matrix(Λ)*M;
  
  /**
   * Among-column covariance.
   */
  V:InverseWishart <- V;

  function rows() -> Integer {
    return N.rows();
  }
  
  function columns() -> Integer {
    return N.columns();
  }

  function simulate() -> Real[_,_] {
    return simulate_matrix_normal_inverse_wishart(N.value(), Λ.value(),
        V.Ψ.value(), V.k.value());
  }
  
  function logpdf(X:Real[_,_]) -> Real {   
    return logpdf_matrix_normal_inverse_wishart(X, N.value(), Λ.value(), V.Ψ.value(), V.k.value());
  }

  function update(X:Real[_,_]) {
    (V.Ψ, V.k) <- update_matrix_normal_inverse_wishart(X, N.value(), Λ.value(), V.Ψ.value(), V.k.value());
  }

  function downdate(X:Real[_,_]) {
    (V.Ψ, V.k) <- downdate_matrix_normal_inverse_wishart(X, N.value(), Λ.value(), V.Ψ.value(), V.k.value());
  }

  function graftMatrixNormalInverseWishart(compare:Distribution<Real[_,_]>) ->
      MatrixNormalInverseWishart? {
    prune();
    if V == compare {
      return this;
    } else {
      return nil;
    }
  }

  function link() {
    V.setChild(this);
  }
  
  function unlink() {
    V.releaseChild();
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "MatrixNormalInverseWishart");
    buffer.set("M", solve(Λ.value(), N.value()));
    buffer.set("Σ", inv(Λ.value()));
    buffer.set("Ψ", V.Ψ.value());
    buffer.set("k", V.k.value());
  }
}

function MatrixNormalInverseWishart(M:Expression<Real[_,_]>,
    U:Expression<Real[_,_]>, V:InverseWishart) -> MatrixNormalInverseWishart {
  m:MatrixNormalInverseWishart(M, U, V);
  m.link();
  return m;
}
