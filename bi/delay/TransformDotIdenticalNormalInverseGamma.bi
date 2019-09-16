/*
 * Dot transformation of a multivariate normal-inverse-gamma random
 * variate.
 */
final class TransformDotIdenticalNormalInverseGamma(a:Real[_],
    x:DelayIdenticalNormalInverseGamma, c:Real) <
    TransformDot<Real>(a, c) {
  /**
   * Random variate.
   */
  x:DelayIdenticalNormalInverseGamma <- x;
}

function TransformDotIdenticalNormalInverseGamma(a:Real[_],
    x:DelayIdenticalNormalInverseGamma, c:Real) ->
    TransformDotIdenticalNormalInverseGamma {
  m:TransformDotIdenticalNormalInverseGamma(a, x, c);
  return m;    
}

function TransformDotIdenticalNormalInverseGamma(a:Real[_],
    x:DelayIdenticalNormalInverseGamma) ->
    TransformDotIdenticalNormalInverseGamma {
  return TransformDotIdenticalNormalInverseGamma(a, x, 0.0);
}
