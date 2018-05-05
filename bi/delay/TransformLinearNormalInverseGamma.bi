/*
 * Linear transformation of a normal-inverse-gamma random variate.
 */
class TransformLinearNormalInverseGamma(a:Real, x:DelayNormalInverseGamma,
    c:Real) < TransformLinear<Real>(a, c) {
  /**
   * Random variate.
   */
  x:DelayNormalInverseGamma <- x;
}

function TransformLinearNormalInverseGamma(a:Real, x:DelayNormalInverseGamma,
    c:Real) -> TransformLinearNormalInverseGamma {
  m:TransformLinearNormalInverseGamma(a, x, c);
  return m;
}
