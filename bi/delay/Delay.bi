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
   * Is this node realized?
   */
  realized:Boolean <- false;
  
  /**
   * Realize.
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
   * Remove the $M$-path child of this node.
   */
  function detach() {
    parent:Delay? <- this.parent;
    if (parent?) {
      parent!.child <- nil;
      parent <- nil;
    }
  }
  
  /**
   * Prune the $M$-path from below this node.
   */
  function prune() {
    if (child?) {
      child!.prune();
      child!.realize();
    }
  }
}
