/*
 * Node states for delayed sampling.
 */
UNINITIALIZED:Integer8 <- 0;
INITIALIZED:Integer8 <- 1;
MARGINALIZED:Integer8 <- 2;
REALIZED:Integer8 <- 3;

/**
 * Interface for delayed sampling of random variables.
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
   * State of the variate.
   */
  state:Integer8 <- UNINITIALIZED;

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
  function initialize();

  /**
   * Marginalize.
   */
  function marginalize();
  
  /**
   * Realize (simulate or observe).
   */
  function realize();

  /**
   * Graft the $M$-path to this node.
   */
  function graft();

  /**
   * Graft the $M$-path to this node.
   *
   * - c: The child node (caller) that will itself be part of the $M$-path.
   */
  function graft(c:Delay);
  
  /**
   * Prune the $M$-path from below this node.
   */
  function prune();
}
