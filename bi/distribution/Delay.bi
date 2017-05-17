import math;
import assert;

/*
 * Node states for delayed sampling.
 */
UNINITIALISED:Integer <- 0;
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
  state:Integer <- UNINITIALISED;
  
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
    return isMarginalised() && !hasChild;
  }

  /**
   * Is this node in the uninitialised state?
   */
  function isUninitialised() -> Boolean {
    return state == UNINITIALISED;
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
   * Is the value of this node missing?
   */
  function isMissing() -> Boolean {
    return missing;
  }
  
  /**
   * Does this node have a deterministic relationship with its parent?
   */
  function isDeterministic() -> Boolean {
    return false;
  }
    
  /**
   * Initialise as a root node.
   */
  function initialize() {
    this.hasParent <- false;
    this.hasChild <- false;
    this.state <- MARGINALISED;
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
    this.state <- INITIALISED;
  }
  
  /**
   * Marginalise the variate.
   */
  function marginalize() {
    assert(isInitialised());
    assert(hasParent);
    
    doMarginalize();
    this.state <- MARGINALISED;
  }
  
  /**
   * Forward sample the variate.
   */
  function forward() {
    assert(isInitialised());
    
    doForward();
    if (isDeterministic()) {
      this.state <- REALISED;
    } else {
      this.state <- MARGINALISED;
    }
  }
  
  /**
   * Realise the variate.
   */
  function realize() {
    assert(isInitialised() || isTerminal());
    
    this.state <- REALISED;
    if (hasParent) {
      parent.removeChild();
    }
    if (missing) {
      doSample();
    } else {
      doObserve();
    }
    if (hasParent && !parent.isRealised()) {
      doCondition();
      if (isDeterministic()) {
        parent.realize();
      }
    }
    removeParent();
  }

  /**
   * Graft the stem to this node.
   */
  function graft() {
    if (isMarginalised()) {
      if (hasChild) {
        child.prune();
        removeChild();
      }
    } else if (isInitialised()) {
      parent.graft(this);
      if (parent.isRealised()) {
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
    assert(isMarginalised());
    
    if (hasChild) {
      child.prune();
      removeChild();
    }
    if (!isRealised()) { // deterministic child may have triggered realisation
      realize();
    }
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
    cast(this)->parent = nullptr;
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
    cast(this)->child = nullptr;
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
    cast(this)->parent = nullptr;
    cast(this)->child = nullptr;
    }}
  }
}
