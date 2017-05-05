import math;
import assert;

/*
 * Node states for delayed sampling.
 */
INITIALISED:Integer <- 1;
MARGINALISED:Integer <- 2;
REALISED:Integer <- 3;

/**
 * Node interface for delayed sampling.
 */
class Delay {
  /**
   * State of the variate.
   */
  state:Integer;
  
  /**
   * Parent, if any.
   */
  parent:Delay;
  
  /**
   * Child, if one exists and it is on the stem.
   */
  child:Delay;
  
  /**
   * Is there a parent?
   */
  hasParent:Boolean;
  
  /**
   * Is there a child?
   */
  hasChild:Boolean;
  
  /**
   * Is this a root node?
   */
  function isRoot() -> Boolean {
    return !hasParent;
  }
  
  /**
   * Is this the terminal node of a stem?
   */
  function isTerminal() -> Boolean {
    return isMarginalised() && !hasChild;
  }
  
  /**
   * Is this node in the initialised state?
   */
  function isInitialised() -> Boolean {
    return state == INITIALISED;
  }

  /**
   * Is this node in the marginalised state?
   */
  function isMarginalised() -> Boolean {
    return state == MARGINALISED;
  }

  /**
   * Is this node in the realised state?
   */
  function isRealised() -> Boolean {
    return state == REALISED;
  }

  /**
   * Initialise as a root node.
   */
  function initialise() {
    this.hasParent <- false;
    this.hasChild <- false;
    this.state <- MARGINALISED;
  }
  
  /**
   * Initialise as a non-root node.
   *
   * `parent` The parent node.
   */
  function initialise(parent:Delay) {
    this.parent <- parent;
    this.hasParent <- true;
    this.hasChild <- false;
    this.state <- INITIALISED;
  }
  
  /**
   * Marginalise the variate.
   */
  function marginalise() {
    assert(isInitialised());
    assert(hasParent);
    
    this.state <- MARGINALISED;
    doMarginalise();
  }
  
  /**
   * Realise the variate.
   */
  function realise() {
    assert(isInitialised() || isTerminal());
    
    this.state <- REALISED;
    if (hasParent) {
      parent.removeChild();
    }
    doRealise();
  } 
  
  /**
   * Sample the value.
   */
  function sample() {
    assert(isTerminal());
    
    doSample();
    realise();
  }

  /**
   * Observe the value.
   */
  function observe() {
    assert(isTerminal());
    
    doObserve();
    realise();
  }

  /**
   * Ensure that the node has a value
   */
  function value() {
    if (!isRealised()) {
      graft();
      sample();
    }
  }

  /**
   * Graft the stem to this node.
   */
  function graft() {
    if (isMarginalised()) {
      if (hasChild) {
        child.prune();
        this.hasChild <- false;
      }
    } else {
      if (!parent.isRealised()) {
        parent.graft(this);
      }
      marginalise();
    }

    assert(isMarginalised());
  }

  /**
   * Graft the stem to this node.
   *
   * `c` The child node that called this, and that will itself be part
   * of the stem.
   */
  function graft(c:Delay) {
    assert(isInitialised() || isMarginalised());
  
    if (isMarginalised()) {
      if (hasChild) {
        child.prune();
        removeChild();
      }
    } else {
      if (!parent.isRealised()) {
        parent.graft(this);
      }
      marginalise();
    }
    setChild(c);
    
    assert(isMarginalised());
  }
  
  /**
   * Prune the stem from below this node.
   */
  function prune() {
    assert(isMarginalised());
    
    if (hasChild) {
      child.prune();
      removeChild();
    }
    sample();
  }

  /**
   * Set the child.
   */
  function setChild(c:Delay) {
    this.child <- c;
    this.hasChild <- true;
  }

  /**
   * Remove the child.
   */
  function removeChild() {
    this.hasChild <- false;
  }
  
  /**
   * Functions to be implemented by derived types
   */
  function doMarginalise() {
    //
  }
  function doRealise() {
    //
  }
  function doSample() {
    //
  }
  function doObserve() {
    //
  }
}
