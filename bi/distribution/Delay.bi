/**
 * Interface for distributions that may appear as nodes on the delayed
 * sampling $M$-path.
 */
abstract class Delay {
  /**
   * Child, if one exists and it is on the $M$-path.
   */
  child:Delay?;
  
  /**
   * Realize a value for the node.
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
