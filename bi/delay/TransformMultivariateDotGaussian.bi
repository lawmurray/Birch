/*
 * Dot transformation of a multivariate Gaussian random variate.
 */
class TransformMultivariateDotGaussian(a:Real[_],
    x:DelayMultivariateGaussian, c:Real) <
    TransformMultivariateDot<Real>(a, c) {
  /**
   * Random variate.
   */
  x:DelayMultivariateGaussian <- x;
}

function TransformMultivariateDotGaussian(a:Real[_],
    x:DelayMultivariateGaussian, c:Real) ->
    TransformMultivariateDotGaussian {
  m:TransformMultivariateDotGaussian(a, x, c);
  return m;    
}

function TransformMultivariateDotGaussian(a:Real[_],
    x:DelayMultivariateGaussian) -> TransformMultivariateDotGaussian {
  return TransformMultivariateDotGaussian(a, x, 0.0);
}
