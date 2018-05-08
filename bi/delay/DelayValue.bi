/*
 * Type-specific interface for delayed sampling $M$-path nodes.
 *
 * - Value: Value type.
 */
class DelayValue<Value> < Delay {
  /**
   * Associated random variable.
   */
  x:Random<Value>?;
    
  function realize() {
    if (parent?) {
      parent!.child <- nil;
      // ^ doing this now makes the parent a terminal node, so that within
      //   simulate() or observe(), realization of the parent can be
      //   forced also; this is useful for deterministic relationships (e.g.
      //   see DelayDelta)
    }
    if (x?) {
      x!.value();
    }
    parent <- nil;
  }

  /**
   * Simulate a random variate.
   *
   * Return: The value.
   */
  function simulate() -> Value {
    assert false;
  }

  /**
   * Observe a random variate.
   *
   * - x: The value.
   *
   * Return: the log likelihood.
   */
  function observe(x:Value) -> Real {
    assert false;
  }

  /**
   * Update the parent given the random variate.
   *
   * - x: The value.
   */
  function condition(x:Value) {
    //
  }

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
