/*
 * Linear transformation of a multivariate normal-inverse-gamma random
 * variate.
 */
final class TransformLinearIdenticalNormalInverseGamma(A:Real[_,_],
    x:DelayIdenticalNormalInverseGamma, c:Real[_]) <
    TransformLinearMultivariate<Real>(A, c) {
  /**
   * Random variate.
   */
  x:DelayIdenticalNormalInverseGamma <- x;
}

function TransformLinearIdenticalNormalInverseGamma(A:Real[_,_],
    x:DelayIdenticalNormalInverseGamma, c:Real[_]) ->
    TransformLinearIdenticalNormalInverseGamma {
  m:TransformLinearIdenticalNormalInverseGamma(A, x, c);
  return m;    
}

function TransformLinearIdenticalNormalInverseGamma(A:Real[_,_],
    x:DelayIdenticalNormalInverseGamma) ->
    TransformLinearIdenticalNormalInverseGamma {
  return TransformLinearIdenticalNormalInverseGamma(A, x,
      vector(0.0, rows(A)));
}
