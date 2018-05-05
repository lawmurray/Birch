/*
 * Linear transformation of a multivariate Gaussian random variate.
 */
class TransformMultivariateLinearGaussian(A:Real[_,_],
    x:DelayMultivariateGaussian, c:Real[_]) <
    TransformMultivariateLinear<Real>(A, c) {
  /**
   * Random variate.
   */
  x:DelayMultivariateGaussian <- x;
}

function TransformMultivariateLinearGaussian(A:Real[_,_],
    x:DelayMultivariateGaussian, c:Real[_]) ->
    TransformMultivariateLinearGaussian {
  m:TransformMultivariateLinearGaussian(A, x, c);
  return m;    
}

function TransformMultivariateLinearGaussian(A:Real[_,_],
    x:DelayMultivariateGaussian) -> TransformMultivariateLinearGaussian {
  return TransformMultivariateLinearGaussian(A, x, vector(0.0, rows(A)));
}
