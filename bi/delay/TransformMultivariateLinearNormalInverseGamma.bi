/*
 * Linear transformation of a multivariate normal-inverse-gamma random
 * variate.
 */
final class TransformMultivariateLinearNormalInverseGamma(A:Real[_,_],
    x:DelayIdenticalNormalInverseGamma, c:Real[_]) <
    TransformMultivariateLinear<Real>(A, c) {
  /**
   * Random variate.
   */
  x:DelayIdenticalNormalInverseGamma <- x;
}

function TransformMultivariateLinearNormalInverseGamma(A:Real[_,_],
    x:DelayIdenticalNormalInverseGamma, c:Real[_]) ->
    TransformMultivariateLinearNormalInverseGamma {
  m:TransformMultivariateLinearNormalInverseGamma(A, x, c);
  return m;    
}

function TransformMultivariateLinearNormalInverseGamma(A:Real[_,_],
    x:DelayIdenticalNormalInverseGamma) ->
    TransformMultivariateLinearNormalInverseGamma {
  return TransformMultivariateLinearNormalInverseGamma(A, x,
      vector(0.0, rows(A)));
}
