/*
 * Grafted matrix normal-inverse-gamma distribution.
 */
final class MatrixNormalInverseGamma(M:Expression<Real[_,_]>,
    Σ:Expression<Real[_,_]>, σ2:IndependentInverseGamma) <
    Distribution<Real[_,_]> {
  /**
   * Precision.
   */
  Λ:Expression<LLT> <- llt(inv(llt(Σ)));

  /**
   * Precision times mean.
   */
  N:Expression<Real[_,_]> <- matrix(Λ)*M;

  /**
   * Variance shapes.
   */
  α:Expression<Real> <- σ2.α;

  /**
   * Variance scale accumulators.
   */
  γ:Expression<Real[_]> <- σ2.β + 0.5*diagonal(transpose(N)*M);

  /**
   * Variance scales.
   */
  σ2:IndependentInverseGamma& <- σ2;

  function rows() -> Integer {
    return N.rows();
  }
  
  function columns() -> Integer {
    return N.columns();
  }

  function simulate() -> Real[_,_] {
    return simulate_matrix_normal_inverse_gamma(N.value(), Λ.value(),
        α.value(), gamma_to_beta(γ.value(), N.value(), Λ.value()));
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    return logpdf_matrix_normal_inverse_gamma(X, N.value(), Λ.value(),
        α.value(), gamma_to_beta(γ.value(), N.value(), Λ.value()));
  }

  function update(X:Real[_,_]) {
    (σ2.α, σ2.β) <- update_matrix_normal_inverse_gamma(X, N.value(),
        Λ.value(), α.value(), gamma_to_beta(γ.value(), N.value(),
        Λ.value()));
  }

  function downdate(X:Real[_,_]) {
    (σ2.α, σ2.β) <- downdate_matrix_normal_inverse_gamma(X, N.value(),
        Λ.value(), α.value(), gamma_to_beta(γ.value(), N.value(),
        Λ.value()));
  }

  function graftMatrixNormalInverseGamma(compare:Distribution<Real[_]>) ->
      MatrixNormalInverseGamma? {
    prune();
    graftFinalize();
    if σ2 == compare {
      return this;
    } else {
      return nil;
    }
  }

  function graftFinalize() -> Boolean {
    Λ.value();
    N.value();
    α.value();
    γ.value();
    if !σ2.hasValue() {
      σ2.setChild(this);
      return true;
    } else {
      return false;
    }
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "MatrixNormalInverseGamma");
    buffer.set("M", solve(Λ.value(), N.value()));
    buffer.set("Σ", inv(Λ));
    buffer.set("α", α);
    buffer.set("β", gamma_to_beta(γ.value(), N.value(), Λ.value()));
  }
}

function MatrixNormalInverseGamma(M:Expression<Real[_,_]>,
    Σ:Expression<Real[_,_]>, σ2:IndependentInverseGamma) ->
    MatrixNormalInverseGamma {
  m:MatrixNormalInverseGamma(M, Σ, σ2);
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
