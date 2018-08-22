/*
 * Scaling of an gamma random variate.
 */
class TransformScaledGamma(a:Real, x:DelayGamma) {
  /**
   * Scale.
   */
  a:Real<- a;
  
  /**
   * Gamma.
   */
  x:DelayGamma <- x;
  
  function multiply(y:Real) {
    a <- a*y;
  }

  function divide(y:Real) {
    a <- a/y;
  }
}

/*
 * Constructor.
 */
function TransformScaledGamma(a:Real, x:DelayGamma) ->
  TransformScaledGamma {
  m:TransformScaledGamma(a, x);
  return m;
}
