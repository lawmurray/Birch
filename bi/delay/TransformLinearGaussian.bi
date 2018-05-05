/*
 * Linear transformation of a Gaussian random variate.
 */
class TransformLinearGaussian(a:Real, x:DelayGaussian, c:Real) <
    TransformLinear<Real>(a, c) {
  /**
   * Random variate.
   */
  x:DelayGaussian <- x;
}

function TransformLinearGaussian(a:Real, x:DelayGaussian, c:Real) ->
    TransformLinearGaussian {
  m:TransformLinearGaussian(a, x, c);
  return m;
}
