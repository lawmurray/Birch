import math;
import assert;

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
  state:Integer;
  
  /**
   * Is the value missing?
   */
  missing:Boolean;
  
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
  
  function constructor() {
    state <- UNINITIALIZED;
    missing <- true;
    hasParent <- false;
    hasChild <- false;
  }
  
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
    return isMarginalized() && !hasChild;
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
   * Initialise as a root node.
   */
  function initialize() {
    this.hasParent <- false;
    this.hasChild <- false;
    this.state <- MARGINALIZED;
  }
  
  /**
   * Initialise as a non-root node.
   *
   * `parent` The parent node.
   */
  function initialize(parent:Delay) {
    this.parent <- parent;
    this.hasParent <- true;
    this.hasChild <- false;
    this.state <- INITIALIZED;
  }
  
  /**
   * Marginalise the variate.
   */
  function marginalize() {
    assert(isInitialized());
    assert(hasParent);
    
    doMarginalize();
    state <- MARGINALIZED;
  }
  
  /**
   * Forward sample the variate.
   */
  function forward() {
    assert(isInitialized());
    
    doForward();
    state <- MARGINALIZED;
  }
  
  /**
   * Realise the variate.
   */
  function realize() {
    assert(isInitialized() || isTerminal());
    
    state <- REALIZED;
    if (hasParent) {
      parent.removeChild();
    }
    if (missing) {
      doSample();
    } else {
      doObserve();
    }
    if (hasParent && !parent.isRealized()) {
      doCondition();
    }
    removeParent();
  }

  /**
   * Graft the stem to this node.
   */
  function graft() {
    if (isMarginalized()) {
      if (hasChild) {
        child.prune();
        removeChild();
      }
    } else if (isInitialized()) {
      parent.graft(this);
      if (parent.isRealized()) {
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
    assert(isMarginalized());
    
    if (hasChild) {
      child.prune();
      removeChild();
    }
    realize();
  }

  /**
   * Set the parent.
   */
  function setParent(u:Delay) {
    child <- u;
    hasChild <- true;
  }

  /**
   * Remove the parent.
   */
  function removeParent() {
    hasParent <- false;
  }

  /**
   * Set the child.
   */
  function setChild(u:Delay) {
    child <- u;
    hasChild <- true;
  }

  /**
   * Remove the child.
   */
  function removeChild() {
    hasChild <- false;
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
  function doSample() {
    //
  }
  function doObserve() {
    //
  }
  function doCondition() {
    //
  }
  
  function copy(o:Delay) {
    state <- o.state;
    missing <- o.missing;    
    hasParent <- false;
    hasChild <- false;
  }
}
