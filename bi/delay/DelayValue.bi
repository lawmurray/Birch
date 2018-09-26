/*
 * Type-specific interface for delayed sampling $M$-path nodes.
 *
 * - Value: Value type.
 *
 * - x: Associated random variate.
 */
class DelayValue<Value>(x:Random<Value>&) < Delay {
  /**
   * Associated random variate, if any.
   */
  x:Random<Value>& <- x;

  /**
   * Instantiate the associated random variate, if it exists, by simulation.
   */
  function realize() {
    y:Random<Value>? <- x;
    if y? {
      assert !(y!.x?);
      y!.x <- simulate();
      condition(y!.x!);
      realized <- true;
    }
    detach();
  }
  
  /**
   * Simulate a random variate.
   *
   * Return: the value.
   */
  function simulate() -> Value;

  /**
   * Observe a random variate.
   *
   * - x: The value.
   *
   * Return: The log likelihood.
   */
  function observe(x:Value) -> Real;

  /**
   * Update the parent node on th $M$-path given the value of this node.
   *
   * - x: The value.
   */
  function condition(x:Value);
  
  /**
   * Evaluate the probability mass function (if it exists) at a value.
   *
   * - x: The value.
   *
   * Return: the probability mass.
   */
  function pmf(x:Value) -> Real {
    assert false;
  }

  /**
   * Evaluate the probability density function (if it exists) at a value.
   *
   * - x: The value.
   *
   * Return: the probability density.
   */
  function pdf(x:Value) -> Real {
    assert false;
  }

  /**
   * Evaluate the cumulative distribution function at a value.
   *
   * - x: The value.
   *
   * Return: the cumulative probability
   */
  function cdf(x:Value) -> Real {
    assert false;
  }
  
  /**
   * Finite lower bound of the support of this node, if any.
   */
  function lower() -> Value? {
    return nil;
  }
  
  /**
   * Finite upper bound of the support of this node, if any.
   */
  function upper() -> Value? {
    return nil;
  }
}
