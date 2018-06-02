/*
 * Linear transformation of a discrete random variate.
 */
class TransformLinearDiscrete(a:Integer, x:DelayDiscrete, c:Integer) <
    TransformLinear<Integer>(a, c) {  
  /**
   * Discrete random variable.
   */
  x:DelayDiscrete <- x;
}

/*
 * Constructor.
 */
function TransformLinearDiscrete(a:Integer, x:DelayDiscrete, c:Integer) ->
  TransformLinearDiscrete {
  assert abs(a) == 1;
  m:TransformLinearDiscrete(a, x, c);
  return m;
}
