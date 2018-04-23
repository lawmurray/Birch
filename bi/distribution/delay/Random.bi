/**
 * Delayed random variate.
 *
 * - Value: Value type.
 */
class Random<Value> < Expression<Value> {
  /**
   * Value assignment.
   */
  operator <- x:Value {
    assert isUninitialized();
    this.x <- x;
    realize();
  }

  /**
   * Optional value assignment.
   */
  operator <- x:Value? {
    assert isUninitialized();
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
   * Is the value of the random variable missing?
   */
  function isMissing() -> Boolean {
    return !(x?);
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
  
  function doRealize() {
    if (isMissing()) {
      x <- doSimulate();
    } else {
      w <- doObserve(x!);
    }
  }
  
  function doSimulate() -> Value {
    assert false;
  }
  
  function doObserve(x:Value) -> Real {
    assert false;
  }
}
