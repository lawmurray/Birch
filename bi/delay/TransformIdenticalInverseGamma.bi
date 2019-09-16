/*
 * Matrix scaling of an inverse-gamma random variate.
 */
final class TransformIdenticalInverseGamma(A:Real[_,_],
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
    A <- transpose(x)*A;
  }
}

function TransformIdenticalInverseGamma(A:Real[_,_],
    σ2:DelayInverseGamma) -> TransformIdenticalInverseGamma {
  m:TransformIdenticalInverseGamma(A, σ2);
  return m;
}
