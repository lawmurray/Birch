/*
 * Stack node.
 */
final class StackNode<Type>(x:Type) {
  x:Type <- x;
  next:StackNode<Type>?;

  /**
   * Get value.
   */
  function getValue() -> Type {
    return x;
  }

  /**
   * Get next node.
   */
  function getNext() -> StackNode<Type>? {
    return next;
  }
}
