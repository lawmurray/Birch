/*
 * Interface for delayed sampling $M$-path nodes.
 */
class Delay {
  /**
   * Parent, if one exists.
   */
  parent:Delay&;
  
  /**
   * Child, if one exists and it is on the $M$-path.
   */
  child:Delay?;
  
  /**
   * Has the node realized a value?
   */
  function hasValue() -> Boolean {
    return false;
  }
  
  /**
   * Realize. If a future value has been given, it is used, otherwise a value
   * is simulated.
   */
  function realize();
  
  /**
   * Set the $M$-path child of this node.
   */
  function setChild(child:Delay) {
    child.parent <- this;
    this.child <- child;
  }
  
  /**
   * Remove this node from the $M$-path
   */
  function detach() {
    parent:Delay? <- this.parent;
    if parent? {
      parent!.child <- nil;
    }
    this.parent <- nil;
    assert !this.child?;
  }
  
  /**
   * Prune the $M$-path from below this node.
   */
  function prune() {
    if child? {
      child!.realize();
    }
    assert !this.child?;
  }
}
