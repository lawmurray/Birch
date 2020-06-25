/**
 * Matrix normal-inverse-Wishart distribution.
 */
final class MatrixNormalInverseWishart(M:Expression<Real[_,_]>,
    U:Expression<LLT>, V:InverseWishart) < Distribution<Real[_,_]> {
  /**
   * Precision.
   */
  Λ:Expression<LLT> <- inv(U);

  /**
   * Precision times mean.
   */
  N:Expression<Real[_,_]> <- Λ*M;
  
  /**
   * Among-column covariance.
   */
  V:InverseWishart& <- V;

  function rows() -> Integer {
    return N.rows();
  }
  
  function columns() -> Integer {
    return N.columns();
  }

  function supportsLazy() -> Boolean {
    return true;
  }

  function simulate() -> Real[_,_] {
    auto V <- this.V;
    return simulate_matrix_normal_inverse_wishart(N.value(), Λ.value(), V.Ψ.value(), V.k.value());
  }

  function simulateLazy() -> Real[_,_]? {
    auto V <- this.V;
    return simulate_matrix_normal_inverse_wishart(N.get(), Λ.get(), V.Ψ.get(), V.k.get());
  }
  
  function logpdf(X:Real[_,_]) -> Real {   
    auto V <- this.V;
    return logpdf_matrix_normal_inverse_wishart(X, N.value(), Λ.value(), V.Ψ.value(), V.k.value());
  }

  function logpdfLazy(X:Expression<Real[_,_]>) -> Expression<Real>? {   
    auto V <- this.V;
    return logpdf_lazy_matrix_normal_inverse_wishart(X, N, Λ, V.Ψ, V.k);
  }

  function update(X:Real[_,_]) {
    auto V <- this.V;
    (V.Ψ, V.k) <- box(update_matrix_normal_inverse_wishart(X, N.value(), Λ.value(), V.Ψ.value(), V.k.value()));
  }

  function updateLazy(X:Expression<Real[_,_]>) {
    auto V <- this.V;
    (V.Ψ, V.k) <- update_lazy_matrix_normal_inverse_wishart(X, N, Λ, V.Ψ, V.k);
  }

  function downdate(X:Real[_,_]) {
    auto V <- this.V;
    (V.Ψ, V.k) <- box(downdate_matrix_normal_inverse_wishart(X, N.value(), Λ.value(), V.Ψ.value(), V.k.value()));
  }

  function graftMatrixNormalInverseWishart(compare:Distribution<LLT>) ->
      MatrixNormalInverseWishart? {
    prune();
    auto V <- this.V;
    if V == compare {
      return this;
    } else {
      return nil;
    }
  }

  function link() {
    auto V <- this.V;
    V.setChild(this);
  }
  
  function unlink() {
    auto V <- this.V;
    V.releaseChild(this);
  }

  function write(buffer:Buffer) {
    auto V <- this.V;
    prune();
    buffer.set("class", "MatrixNormalInverseWishart");
    buffer.set("M", solve(Λ.value(), N.value()));
    buffer.set("Σ", inv(Λ.value()));
    buffer.set("Ψ", V.Ψ.value());
    buffer.set("k", V.k.value());
  }
}

function MatrixNormalInverseWishart(M:Expression<Real[_,_]>,
    U:Expression<LLT>, V:InverseWishart) -> MatrixNormalInverseWishart {
  m:MatrixNormalInverseWishart(M, U, V);
  m.link();
  return m;
}
