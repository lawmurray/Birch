/**
 * Variate with a weight.
 */
class WeightedVariate<Variate>(x:Variate) {
  /**
   * The variate.
   */
  x:Variate <- x;

  /**
   * Log-weight of the variate.
   */
  w:Real <- 0.0;
}
