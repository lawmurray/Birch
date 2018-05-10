/*
 * Type-specific interface for delayed sampling $M$-path nodes.
 *
 * - Value: Value type.
 *
 * - x: Associated random variable.
 */
class DelayValue<Value>(x:Random<Value>&) < Delay {
  /**
   * Associated delayed random variate, if any.
   */
  x:Random<Value>& <- x;

  /**
   * Instantiate the associated delayed random variate.
   */
  function value() {
    /* detach from $M$-path; doing this first makes the parent a terminal
     * node, so that within simulate() or observe(), realization of the
     * parent can be forced also; this is useful for deterministic
     * relationships (e.g. see DelayDelta) */
    if (parent?) {
      parent!.child <- nil;
      parent <- nil;
    }

    y:Random<Value>? <- x;
    if y? {
      y!.x <- simulate();
      condition(y!.x!);
    }
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
}
