/*
 * Affine transformation of a multivariate Gaussian variate.
 */
class TransformMultivariateAffineGaussian(A:Real[_,_],
    x:DelayMultivariateGaussian, c:Real[_]) <
    TransformMultivariateAffine(A, c) {
  /**
   * Random variate.
   */
  x:DelayMultivariateGaussian <- x;
}

function TransformMultivariateAffineGaussian(A:Real[_,_],
    x:DelayMultivariateGaussian, c:Real[_]) ->
    TransformMultivariateAffineGaussian {
  m:TransformMultivariateAffineGaussian(A, x, c);
  return m;    
}
