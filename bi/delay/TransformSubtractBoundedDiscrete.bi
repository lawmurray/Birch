/*
 * Difference of two bounded discrete random variates.
 */
class TransformSubtractBoundedDiscrete(x1:DelayBoundedDiscrete,
    x2:DelayBoundedDiscrete) {  
  /**
   * First bounded discrete random variate.
   */
  x1:DelayBoundedDiscrete <- x1;

  /**
   * Second bounded discrete random variate.
   */
  x2:DelayBoundedDiscrete <- x2;
}

/*
 * Constructor.
 */
function TransformSubtractBoundedDiscrete(x1:DelayBoundedDiscrete,
    x2:DelayBoundedDiscrete) -> TransformSubtractBoundedDiscrete {
  m:TransformSubtractBoundedDiscrete(x1, x2);
  return m;
}
