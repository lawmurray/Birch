import math;

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
   * State of the variate.
   */
  state:Integer <- UNINITIALIZED;
  
  /**
   * Is the value missing?
   */
  missing:Boolean <- true;
  
  /**
   * Parent.
   */
  parent:Delay?;
  
  /**
   * Child, if one exists and it is on the stem.
   */
  child:Delay?;
    
  /**
   * Is this a root node?
   */
  function isRoot() -> Boolean {
    return !(parent?);
  }
  
  /**
   * Is this the terminal node of a stem?
   */
  function isTerminal() -> Boolean {
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
   * Is the value of this node missing?
   */
  function isMissing() -> Boolean {
    return missing;
  }

  /**
   * Initialize as a root node.
   */
  function initialize() {
    this.state <- MARGINALIZED;
  }
  
  /**
   * Initialize as a non-root node.
   *
   * `parent` The parent node.
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
    assert parent?;
    
    doMarginalize();
    state <- MARGINALIZED;
  }
  
  /**
   * Forward simulate the variate.
   */
  function forward() {
    assert isInitialized();
    
    doForward();
    state <- MARGINALIZED;
  }
  
  /**
   * Realize the variate.
   */
  function realize() {
    assert isTerminal();
    
    state <- REALIZED;
    if (parent?) {
      parent!.removeChild();
    }
    if (missing) {
      doSimulate();
    } else {
      doObserve();
    }
    if (parent? && !(parent!.isRealized())) {
      doCondition();
    }
    removeParent();
  }

  /**
   * Graft the stem to this node.
   */
  function graft() {
    if (isMarginalized()) {
      if (child?) {
        child!.prune();
        removeChild();
      }
    } else if (isInitialized()) {
      parent!.graft(this);
      if (parent!.isRealized()) {
        forward();
      } else {
        marginalize();
      }
    }
  }

  /**
   * Graft the stem to this node.
   *
   * `c` The child node that called this, and that will itself be part
   * of the stem.
   */
  function graft(c:Delay) {
    graft();
    setChild(c);
  }
  
  /**
   * Prune the stem from below this node.
   */
  function prune() {
    assert isMarginalized();
    
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
  function doSimulate() {
    //
  }
  function doObserve() {
    //
  }
  function doCondition() {
    //
  }
}
