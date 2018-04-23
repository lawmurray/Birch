/*
 * Node states for delayed sampling.
 */
UNINITIALIZED:Integer8 <- 0;
INITIALIZED:Integer8 <- 1;
MARGINALIZED:Integer8 <- 2;
REALIZED:Integer8 <- 3;

/**
 * Random variate.
 *
 * - Value: Value type.
 */
class Random<Value> < Expression<Value> {
  /**
   * Parent.
   */
  parent:Delay?;
  
  /**
   * Child, if one exists and it is on the $M$-path.
   */
  child:Delay&;
  
  /**
   * Weight.
   */
  w:Real <- 0.0;

  /**
   * State of the variate.
   */
  state:Integer8 <- UNINITIALIZED;

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
   * Is this node in the uninitialized state?
   */
  function isUninitialized() -> Boolean {
    return state == UNINITIALIZED;
  }
  
  /**
   * Is this node in the initialized state?
   */
  function isInitialized() -> Boolean {
    return state == INITIALIZED;
  }

  /**
   * Is this node in the marginalized state?
   */
  function isMarginalized() -> Boolean {
    return state == MARGINALIZED;
  }

  /**
   * Is this node in the realized state?
   */
  function isRealized() -> Boolean {
    return state == REALIZED;
  }

  /**
   * Initialize.
   */
  function initialize() {
    this.state <- INITIALIZED;
  }

  /**
   * Marginalize.
   */
  function marginalize() {
    assert isInitialized();
    
    state <- MARGINALIZED;
    doMarginalize();
  }
  
  /**
   * Realize (simulate or observe).
   */
  function realize() {
    assert !isRealized();
    
    if (isUninitialized()) {
      state <- REALIZED;
    } else {
      graft();
      state <- REALIZED;
      if (parent?) {
        parent!.setChild(nil);
        // ^ doing this now makes the parent a terminal node, so that within
        //   doRealize(), realization of the parent can be forced also for
        //   deterministic relationships (e.g. see Delta class)
      }
      if (isMissing()) {
        x <- doSimulate();
      } else {
        w <- doObserve(x!);
      }
      if (parent?) {
        if (!(parent!.isRealized()) && w > -inf) {
          // ^ conditioning doesn't make sense if the observation is not
          //   within the support
          doCondition();
        }
        setParent(nil);
      }
    }
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
   * Select a parent. This is used for lazy construction of the $M$-path,
   * allowing each node to select its parent only when required.
   */
  function attach() {
    setParent(doParent());
  }
  
  /**
   * Graft the $M$-path to this node.
   */
  function graft() {
    if (isMarginalized()) {
      child:Delay? <- this.child;
      if (child?) {
        child!.prune();
        setChild(nil);
      }
    } else if (isInitialized()) {
      attach();
      if (parent?) {
        parent!.graft(this);
      }
      marginalize();
    }
  }

  /**
   * Graft the $M$-path to this node.
   *
   * - c: The child node (caller) that will itself be part of the $M$-path.
   */
  function graft(c:Delay) {
    graft();
    setChild(c);
  }
  
  /**
   * Prune the $M$-path from below this node.
   */
  function prune() {
    assert isMarginalized();
    
    child:Delay? <- this.child;
    if (child?) {
      child!.prune();
      setChild(nil);
    }
    realize();
  }

  function setChild(child:Delay?) {
    this.child <- child;
  }

  /**
   * Node-specific parent selection.
   */
  function doParent() -> Delay? {
    return nil;
  }
  
  /**
   * Node-specific marginalization.
   */
  function doMarginalize() {
    //
  }
  
  /**
   * Node-specific conditioning.
   */
  function doCondition() {
    assert false;
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
}
