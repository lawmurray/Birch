/**
 * Linear transformation of a discrete random variate.
 */
class TransformLinearDiscrete(a:Integer, x:DelayValue<Integer>, c:Integer) <
    TransformLinear<Integer>(a, c) {  
  /**
   * Binomial.
   */
  x:DelayValue<Integer> <- x;
}

/*
 * Constructor.
 */
function TransformLinearDiscrete(a:Integer, x:DelayValue<Integer>, c:Integer) ->
  TransformLinearDiscrete {
  assert abs(a) == 1;
  m:TransformLinearDiscrete(a, x, c);
  return m;
}
