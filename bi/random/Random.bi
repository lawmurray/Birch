
/**
 * Random variate.
 *
 * - Value: Value type.
 */
class Random<Value> < Expression<Value> {
  /*
   * Associated node on delayed sampling $M$-path.
   */
  delay:DelayValue<Value>?;

  /**
   * Value.
   */
  x:Value?;
  
  /**
   * Weight.
   */
  w:Real?;

  /**
   * Value assignment.
   */
  operator <- x:Value {
    assert isMissing();
    this.x <- x;
    realize();
  }

  /**
   * Optional value assignment.
   */
  operator <- x:Value? {
    assert isMissing();
    if (x?) {
      this.x <- x!;
      realize();
    }
  }

  /**
   * Are the values of any random variables within this expression missing?
   */
  function isMissing() -> Boolean {
    return !x?;
  }

  /**
   * Get the value of the random variable, forcing realization if necessary.
   */
  function value() -> Value {
    if isMissing() {
      realize();
      assert x?;
    }
    return x!;
  }

  /**
   * Simulate the random variable.
   */
  function simulate() -> Value {
    realize();
    assert x?;
    return x!;
  }

  /**
   * Observe the random variable.
   *
   * - x: The observed value.
   *
   * Return: the log likelihood.
   */
  function observe(x:Value) -> Real {
    this.x <- x;
    realize();
    assert w?;
    return w!;
  }

  /**
   * Evaluate the probability mass function (if it exists) at a value.
   *
   * - x: The value.
   *
   * Return: the probability mass.
   */
  function pmf(x:Value) -> Real {
    delay <- graft();
    assert delay?;
    return delay!.pmf(x);
  }

  /**
   * Evaluate the probability density function (if it exists) at a value.
   *
   * - x: The value.
   *
   * Return: the probability density.
   */
  function pdf(x:Value) -> Real {
    delay <- graft();
    assert delay?;
    return delay!.pdf(x);
  }

  /**
   * Evaluate the cumulative distribution function at a value.
   *
   * - x: The value.
   *
   * Return: the cumulative probability
   */
  function cdf(x:Value) -> Real {
    delay <- graft();
    assert delay?;
    return delay!.cdf(x);
  }

  /*
   * Realize a value for this random variable.
   */
  function realize() {
    delay <- graft();
    if (delay?) {
      delay!.realize();
      delay <- nil;
    }
  }
  
  /*
   * Graft this random variable onto the delayed sampling $M$-path.
   *
   * Return: Associated node on delayed sampling $M$-path.
   */
  function graft() -> DelayValue<Value>? {
    return nil;
  }
}
