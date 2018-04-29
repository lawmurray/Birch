/**
 * Type-specific interface for delayed sampling of random variables.
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
      doCondition(y!.x!);
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
