/*
 * Interface for delayed sampling $M$-path nodes.
 */
class Delay {
  /**
   * Parent, if one exists.
   */
  parent:Delay?;
  
  /**
   * Child, if one exists and it is on the $M$-path.
   */
  child:Delay&;
  
  /**
   * Realize by simulation.
   */
  function realize();
  
  /**
   * Set the child of this node, on the $M$-path.
   */
  function setChild(child:Delay) {
    child.parent <- this;
    this.child <- child;
  }
  
  /**
   * Prune the $M$-path from below this node.
   */
  function prune() {
    child:Delay? <- this.child;
    if (child?) {
      child!.prune();
      child!.realize();
    }
  }
}
