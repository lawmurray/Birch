/*
 * Linear transformation of a binomial variate.
 */
class TransformLinearBinomial(a:Integer, x:DelayBinomial, c:Integer) <
    TransformLinear<Integer>(a, c) {  
  /**
   * Binomial.
   */
  x:DelayBinomial <- x;
}

/*
 * Constructor.
 */
function TransformLinearBinomial(a:Integer, x:DelayBinomial, c:Integer) ->
  TransformLinearBinomial {
  assert abs(a) == 1;
  m:TransformLinearBinomial(a, x, c);
  return m;
}
