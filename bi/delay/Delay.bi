/**
 * Interface for delayed sampling $M$-path nodes.
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
   * Realize (simulate or observe).
   */
  function realize();
  
  /**
   * Prune the $M$-path from below this node.
   */
  function prune() {
    child:Delay? <- this.child;
    if (child?) {
      child!.prune();
      child!.realize();
      child <- nil;
    }
  }
}
