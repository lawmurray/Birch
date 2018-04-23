/**
 * Abstract interface for delayed sampling of random variables.
 */
class Delay {
  /**
   * Is this node in the uninitialized state?
   */
  function isUninitialized() -> Boolean {
    assert false;
  }
  
  /**
   * Is this node in the initialized state?
   */
  function isInitialized() -> Boolean {
    assert false;
  }

  /**
   * Is this node in the marginalized state?
   */
  function isMarginalized() -> Boolean {
    assert false;
  }

  /**
   * Is this node in the realized state?
   */
  function isRealized() -> Boolean {
    assert false;
  }

  /**
   * Graft the $M$-path to this node.
   *
   * - c: The child node (caller) that will itself be part of the $M$-path.
   */
  function graft(c:Delay) {
    assert false;
  }
  
  /**
   * Prune the $M$-path from below this node.
   */
  function prune() {
    assert false;
  }

  /**
   * Set parent.
   */
  function setParent(parent:Delay?) {
    assert false;
  }

  /**
   * Set child.
   */
  function setChild(child:Delay?) {
    assert false;
  }
}
