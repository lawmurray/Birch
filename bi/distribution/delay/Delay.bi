/*
 * Node states for delayed sampling.
 */
UNINITIALIZED:Integer <- 0;
INITIALIZED:Integer <- 1;
MARGINALIZED:Integer <- 2;
REALIZED:Integer <- 3;

/**
 * Node interface for delayed sampling.
 */
class Delay {
  /**
   * Parent.
   */
  parent:Delay?;
  
  /**
   * Child, if one exists and it is on the M-path.
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
   * Is this a root node?
   */
  function isRoot() -> Boolean {
    return !(parent?);
  }
  
  /**
   * Is this the terminal node of an M-path?
   */
  function isTerminal() -> Boolean {
    child:Delay? <- this.child;
    return isMarginalized() && !(child?);
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
   * Initialize as a root node.
   */
  function initialize() {
    this.state <- INITIALIZED;
  }
  
  /**
   * Initialize as a non-root node.
   *
   * - parent: The parent node.
   */
  function initialize(parent:Delay) {
    this.parent <- parent;
    this.state <- INITIALIZED;
  }

  /**
   * Marginalize the variate.
   */
  function marginalize() {
    assert isInitialized();
    
    state <- MARGINALIZED;
    if (parent? && parent!.isRealized()) {
      doForward();
    } else {
      doMarginalize();
    }
  }
  
  /**
   * Realize the variate.
   */
  function realize() {
    assert !isRealized();
    
    if (isUninitialized()) {
      state <- REALIZED;
    } else {
      graft();
      state <- REALIZED;
      if (parent?) {
        parent!.removeChild();
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
        removeParent();
      }
    }
  }
  
  /**
   * Graft the M-path to this node.
   */
  function graft() {
    if (isMarginalized()) {
      child:Delay? <- this.child;
      if (child?) {
        child!.prune();
        removeChild();
      }
    } else if (isInitialized()) {
      if (parent?) {
        parent!.graft(this);
      }
      marginalize();
    }
  }

  /**
   * Graft the M-path to this node.
   *
   * - c: The child node (caller) that will itself be part of the M-path.
   */
  function graft(c:Delay) {
    graft();
    setChild(c);
  }
  
  /**
   * Prune the M-path from below this node.
   */
  function prune() {
    assert isMarginalized();
    
    child:Delay? <- this.child;
    if (child?) {
      child!.prune();
      removeChild();
    }
    realize();
  }

  /**
   * Set the parent.
   */
  function setParent(u:Delay) {
    parent <- u;
  }

  /**
   * Remove the parent.
   */
  function removeParent() {
    parent <- nil;
  }

  /**
   * Set the child.
   */
  function setChild(u:Delay) {
    child <- u;
  }

  /**
   * Remove the child.
   */
  function removeChild() {
    child <- nil;
  }
  
  /*
   * Derived type requirements.
   */
  function doMarginalize() {
    //
  }
  function doForward() {
    //
  }
  function doCondition() {
    assert false;
  }
  function doRealize() {
    assert false;
  }
}
