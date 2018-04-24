/**
 * Typed interface for delayed sampling of random variables.
 *
 * - x: Associated random variable.
 */
class DelayValue<Value>(x:Random<Value>) < Delay {
  /**
   * Associated random variable.
   */
  x:Random<Value>& <- x;
  
  /**
   * Weight.
   */
  w:Real <- 0.0;

  function initialize() {
    this.state <- INITIALIZED;
  }

  function marginalize() {
    assert isInitialized();
    
    state <- MARGINALIZED;
    doMarginalize();
  }
  
  function realize() {
    assert !isRealized();
    
    if (isUninitialized()) {
      state <- REALIZED;
    } else {
      graft();
      state <- REALIZED;
      if (parent?) {
        parent!.child <- nil;
        // ^ doing this now makes the parent a terminal node, so that within
        //   doRealize(), realization of the parent can be forced also for
        //   deterministic relationships (e.g. see Delta class)
      }
      
      y:Random<Value>? <- x;
      assert y?;
      if (y!.isMissing()) {
        y!.x <- doSimulate();
      } else {
        w <- doObserve(y!.x!);
      }

      if (parent?) {
        if (!(parent!.isRealized()) && w > -inf) {
          // ^ conditioning doesn't make sense if the observation is not
          //   within the support
          doCondition(y!.x!);
        }
        parent <- nil;
      }
    }
  }

  function graft() {
    if (isMarginalized()) {
      child:Delay? <- this.child;
      if (child?) {
        child!.prune();
        child <- nil;
      }
    } else if (isInitialized()) {
      if (parent?) {
        parent!.graft(this);
      }
      marginalize();
    }
  }

  function graft(c:Delay) {
    graft();
    child <- c;
  }
  
  function prune() {
    assert isMarginalized();
    
    child:Delay? <- this.child;
    if (child?) {
      child!.prune();
      child <- nil;
    }
    realize();
  }
  
  /**
   * Node-specific marginalization.
   */
  function doMarginalize() {
    //
  }
  
  /**
   * Node-specific simulation.
   */
  function doSimulate() -> Value {
    assert false;
  }

  /**
   * Node-specific observation.
   */
  function doObserve(x:Value) -> Real {
    assert false;
  }

  /**
   * Node-specific conditioning.
   */
  function doCondition(x:Value) {
    //
  }
}
