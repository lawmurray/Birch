/**
 * Synonym for MatrixGaussian.
 */
final class MatrixNormal = MatrixGaussian;

/**
 * Create matrix Gaussian distribution.
 */
function Normal(M:Expression<Real[_,_]>, U:Expression<Real[_,_]>,
    V:Expression<Real[_,_]>) -> MatrixNormal {
  return Gaussian(M, U, V);
}

/**
 * Create matrix Gaussian distribution.
 */
function Normal(M:Expression<Real[_,_]>, U:Expression<Real[_,_]>,
    V:Real[_,_]) -> MatrixNormal {
  return Gaussian(M, U, V);
}

/**
 * Create matrix Gaussian distribution.
 */
function Normal(M:Expression<Real[_,_]>, U:Real[_,_],
    V:Expression<Real[_,_]>) -> MatrixNormal {
  return Gaussian(M, U, V);
}

/**
 * Create matrix Gaussian distribution.
 */
function Normal(M:Expression<Real[_,_]>, U:Real[_,_], V:Real[_,_]) ->
      MatrixNormal {
  return Gaussian(M, U, V);
}

/**
 * Create matrix Gaussian distribution.
 */
function Normal(M:Real[_,_], U:Expression<Real[_,_]>,
    V:Expression<Real[_,_]>) -> MatrixNormal {
  return Gaussian(M, U, V);
}

/**
 * Create matrix Gaussian distribution.
 */
function Normal(M:Real[_,_], U:Expression<Real[_,_]>, V:Real[_,_]) ->
    MatrixNormal {
  return Gaussian(M, U, V);
}

/**
 * Create matrix Gaussian distribution.
 */
function Normal(M:Real[_,_], U:Real[_,_], V:Real[_,_]) -> MatrixNormal {
  return Gaussian(M, U, V);
}
