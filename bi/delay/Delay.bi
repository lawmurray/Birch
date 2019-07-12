/*
 * Interface for delayed sampling $M$-path nodes.
 */
class Delay {
  /**
   * Child, if one exists and it is on the $M$-path.
   */
  child:Delay?;
  
  /**
   * Realize. If a future value has been given, it is used, otherwise a value
   * is simulated.
   */
  function realize();
  
  /**
   * Set the $M$-path child of this node.
   */
  function setChild(child:Delay) {
    this.child <- child;
  }

  /**
   * Prune the $M$-path from below this node.
   */
  function prune() {
    if child? {
      child!.realize();
      child <- nil;
    }
  }
}
