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
  
  /**
   * Constructor.
   */
  function construct() {
    this.state <- UNINITIALISED;
    this.missing <- true;
    this.hasParent <- false;
    this.hasChild <- false;
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
    
    doMarginalise();
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
  function realise() {
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
        parent.realise();
      }
    }
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
        marginalise();
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
      realise();
    }
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
  
  /*
   * Derived type requirements.
   */
   function doMarginalise() {
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
}
