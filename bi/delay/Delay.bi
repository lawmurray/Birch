/*
 * Interface for delayed sampling $M$-path nodes.
 */
abstract class Delay {
  /**
   * Child, if one exists and it is on the $M$-path.
   */
  child:Delay?;
  
  /**
   * Realize. If a future value has been given, it is used, otherwise a value
   * is simulated.
   */
  abstract function realize();
  
  /**
   * Set the $M$-path child of this node.
   */
  function setChild(child:Delay) {
    assert !this.child?;
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
