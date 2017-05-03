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
   * Parent on the stem, if any.
   */
  parent:Delay;
  
  /**
   * Child on the stem, if any.
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
   * Is this the terminal node of the stem?
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
   * Initialise as a root variate.
   */
  function initialise() {
    this.hasParent <- false;
    this.hasChild <- false;
    this.state <- MARGINALISED;
  }
  
  /**
   * Initialise as a non-root variate.
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
    assert(hasParent && parent.isTerminal());
    
    this.state <- MARGINALISED;
    doMarginalise();
  }
  
  /**
   * Marginalise the variate.
   *
   * `child` The calling child, which is also being marginalised.
   */
  function marginalise(child:Delay) {
    assert(isInitialised());
    assert(hasParent && parent.isTerminal());
    
    this.child <- child;
    this.hasChild <- true;
    this.state <- MARGINALISED;
    doMarginalise();
  }
  
  /**
   * Realise the variate.
   */
  function realise() {
    assert(isTerminal());
    
    this.state <- REALISED;
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
      }
    } else {
      parent.graft();
      marginalise();
    }
  }
  
  /**
   * Prune the stem from below this node.
   */
  function prune() {
    assert(isMarginalised());
    
    if (hasChild) {
      child.prune();
    }
    sample();
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
