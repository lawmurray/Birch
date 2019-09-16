/*
 * Dot transformation of a multivariate Gaussian random variate.
 */
final class TransformDotMultivariateGaussian(a:Real[_],
    x:DelayMultivariateGaussian, c:Real) <
    TransformDot<Real>(a, c) {
  /**
   * Random variate.
   */
  x:DelayMultivariateGaussian <- x;
}

function TransformDotMultivariateGaussian(a:Real[_],
    x:DelayMultivariateGaussian, c:Real) ->
    TransformDotMultivariateGaussian {
  m:TransformDotMultivariateGaussian(a, x, c);
  return m;    
}

function TransformDotMultivariateGaussian(a:Real[_],
    x:DelayMultivariateGaussian) -> TransformDotMultivariateGaussian {
  return TransformDotMultivariateGaussian(a, x, 0.0);
}
