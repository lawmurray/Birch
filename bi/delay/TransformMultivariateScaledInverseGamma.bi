/*
 * Multivariate scaling of an inverse-gamma random variate.
 */
final class TransformMultivariateScaledInverseGamma(A:Real[_,_],
    σ2:DelayInverseGamma) {
  /**
   * Scale.
   */
  A:Real[_,_] <- A;
  
  /**
   * Inverse gamma.
   */
  σ2:DelayInverseGamma <- σ2;
  
  function leftMultiply(X:Real[_,_]) {
    A <- X*A;
  }

  function leftDot(x:Real[_]) {
    A <- trans(x)*A;
  }
}

function TransformMultivariateScaledInverseGamma(A:Real[_,_],
    σ2:DelayInverseGamma) -> TransformMultivariateScaledInverseGamma {
  m:TransformMultivariateScaledInverseGamma(A, σ2);
  return m;
}
