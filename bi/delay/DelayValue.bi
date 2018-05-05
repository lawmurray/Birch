/*
 * Type-specific interface for delayed sampling $M$-path nodes.
 *
 * - Value: Value type.
 *
 * - x: Associated random variable.
 */
class DelayValue<Value>(x:Random<Value>) < Delay {
  /**
   * Associated random variable.
   */
  x:Random<Value>& <- x;
  
  /**
   * Simulate the random variable.
   */
  function simulate() -> Value {
    y:Random<Value>? <- x;
    assert y?;
    return y!.simulate();
  }

  /**
   * Observe the random variable.
   *
   * - x: The observed value.
   *
   * Return: the log likelihood.
   */
  function observe(x:Value) -> Real {
    y:Random<Value>? <- this.x;
    assert y?;
    return y!.observe(x);
  }
  
  function realize() {
    if (parent?) {
      parent!.child <- nil;
      // ^ doing this now makes the parent a terminal node, so that within
      //   doSimulate() or doObserve(), realization of the parent can be
      //   forced also; this is useful for deterministic relationships (e.g.
      //   see DelayDelta)
    }
      
    y:Random<Value>? <- x;
    assert y?;
    if (y!.isMissing()) {
      y!.x <- doSimulate();
    } else {
      y!.w <- doObserve(y!.x!);
    }

    if (parent?) {
      if (y!.w! != -inf) {
        doCondition(y!.x!);
      }
      parent <- nil;
    }
  }
  
  /**
   * Node-specific simulate.
   */
  function doSimulate() -> Value {
    assert false;
  }

  /**
   * Node-specific observe.
   */
  function doObserve(x:Value) -> Real {
    assert false;
  }

  /**
   * Node-specific condition.
   */
  function doCondition(x:Value) {
    //
  }
}
