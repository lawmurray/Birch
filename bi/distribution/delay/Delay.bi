/*
 * Node states for delayed sampling.
 */
UNINITIALIZED:Integer <- 0;
INITIALIZED:Integer <- 1;
MARGINALIZED:Integer <- 2;
REALIZED:Integer <- 3;

/**
 * Delayed evaluation functionality.
 */
class Delay {
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
  state:Integer <- UNINITIALIZED;

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
        parent!.child <- nil;
        // ^ doing this now makes the parent a terminal node, so that within
        //   doRealize(), realization of the parent can be forced also for
        //   deterministic relationships (e.g. see Delta class)
      }
      doRealize();
      if (parent?) {
        if (!(parent!.isRealized()) && w > -inf) {
          // ^ conditioning doesn't make sense if the observation is not
          //   within the support
          doCondition();
        }
        parent <- nil;
      }
    }
  }
  
  /**
   * Select a parent. This is used for lazy construction of the $M$-path,
   * allowing each node to select its parent only when required.
   */
  function attach() {
    parent <- doParent();
  }
  
  /**
   * Graft the $M$-path to this node.
   */
  function graft() {
    if (isMarginalized()) {
      child:Delay? <- this.child;
      if (child?) {
        child!.prune();
        child <- nil;
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
    child <- c;
  }
  
  /**
   * Prune the $M$-path from below this node.
   */
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
   * Node-specific realization.
   */
  function doRealize() {
    assert false;
  }
}
