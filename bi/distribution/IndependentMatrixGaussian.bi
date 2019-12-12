/**
 * Matrix Gaussian distribution where each element is independent.
 */
final class IndependentMatrixGaussian(M:Expression<Real[_,_]>,
    v:Expression<Real[_]>) < Distribution<Real[_,_]> {
  /**
   * Mean.
   */
  M:Expression<Real[_,_]> <- M;

  /**
   * Among-column variances.
   */
  σ2:Expression<Real[_]> <- v;

  function rows() -> Integer {
    return M.rows();
  }

  function columns() -> Integer {
    return M.columns();
  }

  function graft(child:Delay?) {
    if delay? {
      delay!.prune();
    } else {
      s1:DelayIndependentInverseGamma?;
      m1:TransformLinearMatrix<DelayMatrixNormalInverseGamma>?;
      m2:DelayMatrixNormalInverseGamma?;

      if (m1 <- M.graftLinearMatrixNormalInverseGamma(child))? && m1!.X.σ2 == σ2.getDelay() {
        delay <- DelayLinearMatrixNormalInverseGammaMatrixGaussian(future, futureUpdate, m1!.A, m1!.X, m1!.C);
      } else if (m2 <- M.graftMatrixNormalInverseGamma(child))? && m2!.σ2 == σ2.getDelay() {
        delay <- DelayMatrixNormalInverseGammaMatrixGaussian(future, futureUpdate, m2!);
      } else if (s1 <- σ2.graftIndependentInverseGamma(child))? {
        delay <- DelayMatrixNormalInverseGamma(future, futureUpdate, M, identity(M.rows()), s1!);
      } else {
        delay <- DelayMatrixGaussian(future, futureUpdate, M, identity(M.rows()), diagonal(σ2));
      }
    }
  }

  function graftMatrixGaussian(child:Delay?) -> DelayMatrixGaussian? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayMatrixGaussian(future, futureUpdate, M,
          identity(M.rows()), diagonal(σ2.value()));
    }
    return DelayMatrixGaussian?(delay);
  }

  function graftMatrixNormalInverseGamma(child:Delay?) -> DelayMatrixNormalInverseGamma? {
    if delay? {
      delay!.prune();
    } else {
      s1:DelayIndependentInverseGamma?;
      if (s1 <- σ2.graftIndependentInverseGamma(child))? {
        delay <- DelayMatrixNormalInverseGamma(future, futureUpdate, M,
            identity(M.rows()), s1!);
      }
    }
    return DelayMatrixNormalInverseGamma?(delay);
  }
}

/**
 * Create matrix Gaussian distribution where each element is independent.
 */
function Gaussian(M:Expression<Real[_,_]>, σ2:Expression<Real[_]>) ->
    IndependentMatrixGaussian {
  m:IndependentMatrixGaussian(M, σ2);
  return m;
}

/**
 * Create matrix Gaussian distribution where each element is independent.
 */
function Gaussian(M:Expression<Real[_,_]>, σ2:Real[_]) ->
    IndependentMatrixGaussian {
  return Gaussian(M, Boxed(σ2));
}

/**
 * Create matrix Gaussian distribution where each element is independent.
 */
function Gaussian(M:Real[_,_], σ2:Expression<Real[_]>) ->
    IndependentMatrixGaussian {
  return Gaussian(Boxed(M), σ2);
}

/**
 * Create matrix Gaussian distribution where each element is independent.
 */
function Gaussian(M:Real[_,_], σ2:Real[_]) -> IndependentMatrixGaussian {
  return Gaussian(Boxed(M), Boxed(σ2));
}
