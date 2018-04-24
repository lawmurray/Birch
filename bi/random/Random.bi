
/**
 * Random variate.
 *
 * - Value: Value type.
 */
class Random<Value> < Expression<Value> {
  /**
   * Associated node in delayed sampling $M$-path.
   */
  delay:DelayValue<Value>?;

  /**
   * Value assignment.
   */
  operator <- x:Value {
    this.x <- x;
    realize();
  }

  /**
   * Optional value assignment.
   */
  operator <- x:Value? {
    if (x?) {
      this.x <- x;
      realize();
    }
  }
  
  /**
   * Get the value of the random variable, forcing its instantiation if
   * it has not already been instantiated.
   */
  function value() -> Value {
    if (isMissing()) {
      realize();
    }
    assert x?;
    return x!;
  }

  /**
   * Simulate the random variable.
   */
  function simulate() -> Value {
    realize();
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
    realize();
    return w;
  }

  /**
   * Graft this random variable into an $M$-path for delayed sampling.
   */
  function graft();

  function doSimulate() -> Integer {
    assert delay?;
    return delay!.doSimulate();
  }
  
  function doObserve(x:Integer) -> Real {
    assert delay?;
    return delay!.doObserve(x);
  }

  function doCondition(x:Integer) {
    assert delay?;
    delay!.doCondition(x);
  }}
