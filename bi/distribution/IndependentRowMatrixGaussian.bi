/**
 * Matrix Gaussian distribution where each row is independent.
 */
final class IndependentRowMatrixGaussian(M:Expression<Real[_,_]>,
    V:Expression<Real[_,_]>) < Distribution<Real[_,_]> {
  /**
   * Mean.
   */
  M:Expression<Real[_,_]> <- M;
  
  /**
   * Among-column covariance.
   */
  V:Expression<Real[_,_]> <- V;
    
  function graft() {
    if delay? {
      delay!.prune();
    } else {
      s1:DelayInverseWishart?;
      m1:TransformLinearMatrix<DelayMatrixNormalInverseWishart>?;
      m2:DelayMatrixNormalInverseWishart?;

      if (m1 <- M.graftLinearMatrixNormalInverseWishart())? && m1!.X.V == V.getDelay() {
        delay <- DelayLinearMatrixNormalInverseWishartMatrixGaussian(future, futureUpdate, m1!.A, m1!.X, m1!.C);
      } else if (m2 <- M.graftMatrixNormalInverseWishart())? && m2!.V == V.getDelay() {
        delay <- DelayMatrixNormalInverseWishartMatrixGaussian(future, futureUpdate, m2!);
      } else if (s1 <- V.graftInverseWishart())? {
        delay <- DelayMatrixNormalInverseWishart(future, futureUpdate, M, identity(M.rows()), s1!);
      } else {
        delay <- DelayMatrixGaussian(future, futureUpdate, M, identity(M.rows()), V);
      }
    }
  }

  function graftMatrixGaussian() -> DelayMatrixGaussian? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayMatrixGaussian(future, futureUpdate, M,
          identity(M.rows()), V);
    }
    return DelayMatrixGaussian?(delay);
  }

  function graftMatrixNormalInverseWishart() -> DelayMatrixNormalInverseWishart? {
    if delay? {
      delay!.prune();
    } else {
      s1:DelayInverseWishart?;
      if (s1 <- V.graftInverseWishart())? {
        delay <- DelayMatrixNormalInverseWishart(future, futureUpdate, M,
            identity(M.rows()), s1!);
      }
    }
    return DelayMatrixNormalInverseWishart?(delay);
  }
}

/**
 * Create matrix Gaussian distribution where each row is independent.
 */
function Gaussian(M:Expression<Real[_,_]>, V:Expression<Real[_,_]>) ->
    IndependentRowMatrixGaussian {
  m:IndependentRowMatrixGaussian(M, V);
  return m;
}

/**
 * Create matrix Gaussian distribution where each row is independent.
 */
function Gaussian(M:Expression<Real[_,_]>, V:Real[_,_]) ->
    IndependentRowMatrixGaussian {
  return Gaussian(M, Boxed(V));
}

/**
 * Create matrix Gaussian distribution where each row is independent.
 */
function Gaussian(M:Real[_,_], V:Expression<Real[_,_]>) ->
    IndependentRowMatrixGaussian {
  return Gaussian(Boxed(M), V);
}
