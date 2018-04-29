/*
 * Affine transformation of a multivariate normal-inverse-gamma variate.
 */
class TransformMultivariateAffineNormalInverseGamma(A:Real[_,_],
    x:DelayMultivariateNormalInverseGamma, c:Real[_]) <
    TransformMultivariateAffine(A, c) {
  /**
   * Random variate.
   */
  x:DelayMultivariateNormalInverseGamma <- x;
}

function TransformMultivariateAffineNormalInverseGamma(A:Real[_,_],
    x:DelayMultivariateNormalInverseGamma, c:Real[_]) ->
    TransformMultivariateAffineNormalInverseGamma {
  m:TransformMultivariateAffineNormalInverseGamma(A, x, c);
  return m;    
}
