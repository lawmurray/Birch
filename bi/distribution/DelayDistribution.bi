/**
 * Value-agnostic interface for distributions. Provides essential
 * functionality for *delayed sampling*, an implementation of automatic
 * marginalization (variable elimination, Rao--Blackwellization) and
 * automatic conjugacy.
 */
abstract class DelayDistribution {
  /**
   * Child, if one exists and it is on the $M$-path.
   */
  child:DelayDistribution&?;
  
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
  final function setChild(child:DelayDistribution) {
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
