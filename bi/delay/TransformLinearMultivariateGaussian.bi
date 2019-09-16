/*
 * Linear transformation of a multivariate Gaussian random variate.
 */
final class TransformLinearMultivariateGaussian(A:Real[_,_],
    x:DelayMultivariateGaussian, c:Real[_]) <
    TransformLinearMultivariate<Real>(A, c) {
  /**
   * Random variate.
   */
  x:DelayMultivariateGaussian <- x;
}

function TransformLinearMultivariateGaussian(A:Real[_,_],
    x:DelayMultivariateGaussian, c:Real[_]) ->
    TransformLinearMultivariateGaussian {
  m:TransformLinearMultivariateGaussian(A, x, c);
  return m;    
}

function TransformLinearMultivariateGaussian(A:Real[_,_],
    x:DelayMultivariateGaussian) -> TransformLinearMultivariateGaussian {
  return TransformLinearMultivariateGaussian(A, x, vector(0.0, rows(A)));
}
