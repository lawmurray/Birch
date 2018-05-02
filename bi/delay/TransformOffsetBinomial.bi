/*
 *
 */
class TransformOffsetBinomial(x:DelayBinomial, c:Integer) {  
  /**
   * Binomial.
   */
  x:DelayBinomial <- x;
  
  /**
   * Offset.
   */
  c:Integer <- c;

  function add(d:Integer) {
    c <- c + d;
  }
}

/*
 * Constructor.
 */
function TransformOffsetBinomial(x:DelayBinomial, c:Integer) ->
  TransformOffsetBinomial {
  m:TransformOffsetBinomial(x, c);
  return m;
}
