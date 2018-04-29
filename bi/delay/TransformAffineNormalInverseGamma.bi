/*
 * Affine transformation of a normal-inverse-gamma variate.
 */
class TransformAffineNormalInverseGamma(a:Real, x:DelayNormalInverseGamma,
    c:Real) < TransformAffine(a, c) {
  /**
   * Random variate.
   */
  x:DelayNormalInverseGamma <- x;
}

function TransformAffineNormalInverseGamma(a:Real, x:DelayNormalInverseGamma,
    c:Real) -> TransformAffineNormalInverseGamma {
  m:TransformAffineNormalInverseGamma(a, x, c);
  return m;
}
