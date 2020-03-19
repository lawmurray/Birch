/*
 * Grafted matrix Gaussian distribution.
 */
class GraftedMatrixGaussian(M:Expression<Real[_,_]>, U:Expression<Real[_,_]>,
    V:Expression<Real[_,_]>) < MatrixGaussian(M, U, V) {
  function graft() -> Distribution<Real[_,_]> {
    if !hasValue() {
      prune();
      graftFinalize();
    }
    return this;
  }

  function graftMatrixGaussian() -> MatrixGaussian? {
    if !hasValue() {
      prune();
      graftFinalize();
      return this;
    } else {
      return nil;
    }
  }

  function graftFinalize() -> Boolean {
    M.value();
    U.value();
    V.value();
    return true;
  }
}

function GraftedMatrixGaussian(M:Expression<Real[_,_]>,
    U:Expression<Real[_,_]>, V:Expression<Real[_,_]>) -> MatrixGaussian {
  m:MatrixGaussian(M, U, V);
  return m;
}
