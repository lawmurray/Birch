/*
 * Type-specific interface for delayed sampling $M$-path nodes.
 *
 * - Value: Value type.
 *
 * - future: Future value.
 * - futureUpdate: When realized, should the future value trigger an
 *   update? (Otherwise a downdate.)
 */
class DelayValue<Value>(future:Value?, futureUpdate:Boolean) < Delay {
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
   * Realize a value for a random variate associated with this node,
   * updating (or downdating) the delayed sampling graph accordingly.
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
      detach();
    }
    return x!;
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
      if futureUpdate {
        update(x);
      } else {
        downdate(x);
      }
    }
    detach();
    return w;
  }

  function realize() {
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
  function logpdf(x:Value) -> Real;

  /**
   * Update the parent node on the $M$-path given the value of this node.
   *
   * - x: The value.
   */
  function update(x:Value) {
    error("update unsupported here");
  }

  /**
   * Downdate the parent node on the $M$-path given the value of this node.
   *
   * - x: The value.
   */
  function downdate(x:Value) {
    error("downdate unsupported here");
  }
  
  /**
   * Evaluate the probability density (or mass) function, if it exists.
   *
   * - x: The value.
   *
   * Return: the probability density (or mass).
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
