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
  state:Integer <- UNINITIALIZED;
  
  /**
   * Is the value missing?
   */
  missing:Boolean <- true;
  
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
  hasParent:Boolean <- false;
  
  /**
   * Is there a child?
   */
  hasChild:Boolean <- false;
  
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
    this.state <- MARGINALIZED;
  }
  
  /**
   * Forward sample the variate.
   */
  function forward() {
    assert(isInitialized());
    
    doForward();
    this.state <- MARGINALIZED;
  }
  
  /**
   * Realise the variate.
   */
  function realize() {
    assert(isInitialized() || isTerminal());
    
    this.state <- REALIZED;
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
    this.child <- u;
    this.hasChild <- true;
  }

  /**
   * Remove the parent.
   */
  function removeParent() {
    cpp{{
    this->parent = nullptr;
    }}
    this.hasParent <- false;
  }

  /**
   * Set the child.
   */
  function setChild(u:Delay) {
    this.child <- u;
    this.hasChild <- true;
  }

  /**
   * Remove the child.
   */
  function removeChild() {
    cpp{{
    this->child = nullptr;
    }}
    this.hasChild <- false;
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
    this.state <- o.state;
    this.missing <- o.missing;    
    this.hasParent <- false;
    this.hasChild <- false;
    cpp{{
    this->parent = nullptr;
    this->child = nullptr;
    }}
  }
}
