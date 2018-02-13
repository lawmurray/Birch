/**
 * Stack node.
 */
class StackNode<Type>(x:Type) {
  x:Type <- x;
  next:StackNode<Type>?;
}
