/*
 * Dot transformation of a multivariate normal-inverse-gamma random
 * variate.
 */
final class TransformMultivariateDotNormalInverseGamma(a:Real[_],
    x:DelayIdenticalNormalInverseGamma, c:Real) <
    TransformMultivariateDot<Real>(a, c) {
  /**
   * Random variate.
   */
  x:DelayIdenticalNormalInverseGamma <- x;
}

function TransformMultivariateDotNormalInverseGamma(a:Real[_],
    x:DelayIdenticalNormalInverseGamma, c:Real) ->
    TransformMultivariateDotNormalInverseGamma {
  m:TransformMultivariateDotNormalInverseGamma(a, x, c);
  return m;    
}

function TransformMultivariateDotNormalInverseGamma(a:Real[_],
    x:DelayIdenticalNormalInverseGamma) ->
    TransformMultivariateDotNormalInverseGamma {
  return TransformMultivariateDotNormalInverseGamma(a, x, 0.0);
}
