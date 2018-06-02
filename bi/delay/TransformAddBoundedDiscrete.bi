/*
 * Sum of two bounded discrete random variates.
 */
class TransformAddBoundedDiscrete(x1:DelayBoundedDiscrete,
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
function TransformAddBoundedDiscrete(x1:DelayBoundedDiscrete,
    x2:DelayBoundedDiscrete) -> TransformAddBoundedDiscrete {
  m:TransformAddBoundedDiscrete(x1, x2);
  return m;
}
