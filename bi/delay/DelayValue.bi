/*
 * Type-specific interface for delayed sampling $M$-path nodes.
 *
 * - Value: Value type.
 *
 * - future: Future value.
 * - futureUpdate: When realized, should the future value trigger an
 *   update? (Otherwise a downdate.)
 */
abstract class DelayValue<Value>(future:Value?, futureUpdate:Boolean) < Delay {
  /**
   * Realized value.
   */
  x:Value?;

  /**
   * Future value. This is set for situations where delayed sampling
   * is used, but when ultimately realized, a particular value (this one)
   * should be assigned, and updates or downdates applied accordingly. It
   * is typically used when replaying traces.
   */
  future:Value? <- future;

  /**
   * When assigned, should the future value trigger an update? (Otherwise
   * a downdate.)
   */
  futureUpdate:Boolean <- futureUpdate;

  /**
   * Does the node have a value?
   */
  function hasValue() -> Boolean {
    return x?;
  }

  /**
   * Realize a value for a random variate associated with this node.
   */
  function value() -> Value {
    if !x? {
      prune();
      if future? {
        x <- future!;
      } else {
        x <- simulate();
      }
      if futureUpdate {
        update(x!);
      } else {
        downdate(x!);
      }
    }
    return x!;
  }

  /**
   * Propose a value for a random variate associated with this node.
   */
  function propose() -> Value {
    if !x? {
      prune();
      if future? {
        x <- future!;
      } else {
        x <- simulate();
      }
      if futureUpdate {
        update(x!);
      } else {
        downdate(x!);
      }
    }
    return x!;
  }

  /**
   * Set a value for a random variate associated with this node,
   * updating the delayed sampling graph accordingly.
   */
  function set(x:Value) {
    assert !this.x?;
    assert !this.future?;

    prune();
    this.x <- x;
    auto w <- logpdf(x);
    ///@todo Would be good to skip evaluation of the logpdf, currently needed
    ///      to ensure an update happens only if valid
    if w > -inf {
      update(x);
    }
  }

  /**
   * Set a value for a random variate associated with this node,
   * downdating the delayed sampling graph accordingly.
   */
  function setWithDowndate(x:Value) {
    assert !this.x?;
    assert !this.future?;

    prune();
    this.x <- x;
    auto w <- logpdf(x);
    ///@todo Would be good to skip evaluation of the logpdf, currently needed
    ///      to ensure a downdate happens only if valid
    if w > -inf {
      downdate(x);
    }
  }
  
  /**
   * Observe a value for a random variate associated with this node,
   * updating (or downdating) the delayed sampling graph accordingly, and
   * returning a weight giving the log pdf (or pmf) of that variate under the
   * distribution.
   */
  function observe(x:Value) -> Real {
    assert !this.x?;
    assert !this.future?;

    prune();
    this.x <- x;
    auto w <- logpdf(x);
    if w > -inf {
      update(x);
    }
    return w;
  }

  /**
   * Observe a value for a random variate associated with this node,
   * updating (or downdating) the delayed sampling graph accordingly, and
   * returning a weight giving the log pdf (or pmf) of that variate under the
   * distribution.
   */
  function observeWithDowndate(x:Value) -> Real {
    assert !this.x?;
    assert !this.future?;

    prune();
    this.x <- x;
    auto w <- logpdf(x);
    if w > -inf {
      downdate(x);
    }
    return w;
  }

  function realize() {
    if x? {
      // nothing to do
    } else {
      prune();
      if future? {
        x <- future!;
      } else {
        x <- simulate();
      }
      if futureUpdate {
        update(x!);
      } else {
        downdate(x!);
      }
    }
  }
  
  /**
   * Simulate a random variate.
   *
   * Return: the value.
   */
  abstract function simulate() -> Value;

  /**
   * Observe a random variate.
   *
   * - x: The value.
   *
   * Return: The log likelihood.
   */
  abstract function logpdf(x:Value) -> Real;

  /**
   * Lazily observe a random variate, if supported.
   *
   * - x: The value.
   *
   * Return: The log likelihood.
   */
  function logpdf(x:Expression<Value>) -> Expression<Real>? {
    return nil;
  }

  /**
   * Update the parent node on the $M$-path given the value of this node.
   *
   * - x: The value.
   */
  function update(x:Value) {
    //
  }

  /**
   * Downdate the parent node on the $M$-path given the value of this node.
   *
   * - x: The value.
   */
  function downdate(x:Value) {
    //
  }
  
  /**
   * Evaluate the probability density (or mass) function, if it exists.
   *
   * - x: The value.
   *
   * Return: the probability density (or mass).
   */
  function pdf(x:Value) -> Real {
    return exp(logpdf(x));
  }

  /**
   * Evaluate the cumulative distribution function at a value.
   *
   * - x: The value.
   *
   * Return: the cumulative probability, if supported.
   */
  function cdf(x:Value) -> Real? {
    return nil;
  }

  /**
   * Evaluate the quantile function at a cumulative probability.
   *
   * - x: The cumulative probability.
   *
   * Return: the quantile, if supported.
   */
  function quantile(p:Real) -> Value? {
    return nil;
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

  function write(buffer:Buffer) {
    buffer.set(value());
  }
}
