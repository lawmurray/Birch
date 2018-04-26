
/**
 * Random variate.
 *
 * - Value: Value type.
 */
class Random<Value> < Expression<Value> {
  /**
   * Value.
   */
  x:Value?;

  /**
   * Associated node in delayed sampling $M$-path.
   */
  delay:DelayValue<Value>?;

  /**
   * Value assignment.
   */
  operator <- x:Value {
    this.x <- x;
    if (delay?) {
      delay!.realize();
    }
  }

  /**
   * Optional value assignment.
   */
  operator <- x:Value? {
    if (x?) {
      this.x <- x;
      if (delay?) {
        delay!.realize();
      }
    }
  }

  /**
   * Are the values of any random variables within this expression missing?
   */
  function isMissing() -> Boolean {
    return !x?;
  }
  
  /**
   * Get the value of the random variable, forcing its instantiation if
   * it has not already been instantiated.
   */
  function value() -> Value {
    if (isMissing() && delay?) {
      delay!.realize();
    }
    assert x?;
    return x!;
  }

  /**
   * Simulate the random variable.
   */
  function simulate() -> Value {
    if (delay?) {
      delay!.realize();
    }
    return x!;
  }
  
  /**
   * Observe the random variable.
   *
   * - x: The observed value.
   *
   * Returns: the log likelihood.
   */
  function observe(x:Value) -> Real {
    this.x <- x;
    if (delay?) {
      delay!.realize();
    }
    return delay!.w;
  }

  /**
   * Graft this random variable into an $M$-path for delayed sampling.
   */
  function graft();

  function doSimulate() -> Value {
    assert delay?;
    return delay!.doSimulate();
  }
  
  function doObserve(x:Value) -> Real {
    assert delay?;
    return delay!.doObserve(x);
  }

  function doCondition(x:Value) {
    assert delay?;
    delay!.doCondition(x);
  }
}
