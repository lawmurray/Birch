/**
 * Interface for distributions that may appear as nodes on the delayed
 * sampling $M$-path.
 */
abstract class Delay {
  /**
   * Child, if one exists and it is on the $M$-path.
   */
  child:Delay&?;
  
  /**
   * Realize a value for the node.
   */
  abstract function realize();

  /**
   * Prune the $M$-path from below this node.
   */
  final function prune() {
    if this.child? {
      auto child <- this.child!;
      child.realize();
    }
  }

  /**
   * Set the $M$-path child of this node. This is used internally by the
   * `link()` member function of the child node.
   */
  final function setChild(child:Delay) {
    assert !this.child? || this.child! == child;
    this.child <- child;
  }

  /**
   * Release the $M$-path child of this node. This is used internally by the
   * `unlink()` member function of the child node.
   */
  final function releaseChild() {
    this.child <- nil;
  }

  /**
   * Establish links with the parent node on the $M$-path.
   */
  function link() {
    //
  }
  
  /**
   * Remove links with the parent node on the $M$-path.
   */
  function unlink() {
    //
  }
}
