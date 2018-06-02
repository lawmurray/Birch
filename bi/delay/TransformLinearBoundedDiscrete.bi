/*
 * Linear transformation of a bounded discrete random variate.
 */
class TransformLinearBoundedDiscrete(a:Integer, x:DelayBoundedDiscrete,
    c:Integer) < TransformLinear<Integer>(a, c) {  
  /**
   * Discrete bounded random variable.
   */
  x:DelayBoundedDiscrete <- x;
}

/*
 * Constructor.
 */
function TransformLinearBoundedDiscrete(a:Integer, x:DelayBoundedDiscrete,
    c:Integer) -> TransformLinearBoundedDiscrete {
  assert abs(a) == 1;
  m:TransformLinearBoundedDiscrete(a, x, c);
  return m;
}
