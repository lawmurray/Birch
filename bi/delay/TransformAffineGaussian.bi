/*
 * Affine transformation of a Gaussian variate.
 */
class TransformAffineGaussian(a:Real, x:DelayGaussian, c:Real) <
    TransformAffine(a, c) {
  /**
   * Random variate.
   */
  x:DelayGaussian <- x;
}

function TransformAffineGaussian(a:Real, x:DelayGaussian, c:Real) ->
    TransformAffineGaussian {
  m:TransformAffineGaussian(a, x, c);
  return m;
}
