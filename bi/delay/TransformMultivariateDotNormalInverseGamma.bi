/*
 * Dot transformation of a multivariate normal-inverse-gamma random
 * variate.
 */
final class TransformMultivariateDotNormalInverseGamma(a:Real[_],
    x:DelayMultivariateNormalInverseGamma, c:Real) <
    TransformMultivariateDot<Real>(a, c) {
  /**
   * Random variate.
   */
  x:DelayMultivariateNormalInverseGamma <- x;
}

function TransformMultivariateDotNormalInverseGamma(a:Real[_],
    x:DelayMultivariateNormalInverseGamma, c:Real) ->
    TransformMultivariateDotNormalInverseGamma {
  m:TransformMultivariateDotNormalInverseGamma(a, x, c);
  return m;    
}

function TransformMultivariateDotNormalInverseGamma(a:Real[_],
    x:DelayMultivariateNormalInverseGamma) ->
    TransformMultivariateDotNormalInverseGamma {
  return TransformMultivariateDotNormalInverseGamma(a, x, 0.0);
}
