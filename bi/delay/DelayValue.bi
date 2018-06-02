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
   * Instantiate the associated delayed random variate by simulating.
   */
  function realize() {
    /* detach from $M$-path; doing this first makes the parent a terminal
     * node, so that within simulate() or observe(), realization of the
     * parent can be forced also; this is useful for deterministic
     * relationships (e.g. see DelayDelta) */
    if (parent?) {
      parent!.child <- nil;
      parent <- nil;
    }

    x:Value <- simulate();
    y:Random<Value>? <- this.x;
    if y? {
      assert !(y!.x?);
      y!.x <- x;
    }
    condition(x);
  }
  
  /**
   * Instantiate the associated delayed random variate an observation.
   *
   * - x: The value.
   *
   * Return: The log likelihood.
   */
  function realize(x:Value) -> Real {
    if (parent?) {
      parent!.child <- nil;
      parent <- nil;
    }

    y:Random<Value>? <- this.x;
    if y? {
      assert !(y!.x?);
      y!.x <- x;
    }
    w:Real <- observe(x);
    condition(x);
    return w;
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
