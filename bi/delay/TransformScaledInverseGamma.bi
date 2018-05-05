/*
 * Scaling of an inverse-gamma random variate.
 */
class TransformScaledInverseGamma(a2:Real, σ2:DelayInverseGamma) {
  /**
   * Scale.
   */
  a2:Real<- a2;
  
  /**
   * Inverse gamma.
   */
  σ2:DelayInverseGamma <- σ2;
  
  function multiply(x2:Real) {
    a2 <- a2*x2;
  }

  function divide(x2:Real) {
    a2 <- a2/x2;
  }
}

/*
 * Constructor.
 */
function TransformScaledInverseGamma(a2:Real, σ2:DelayInverseGamma) ->
  TransformScaledInverseGamma {
  m:TransformScaledInverseGamma(a2, σ2);
  return m;
}
