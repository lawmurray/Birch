/*
 * Stack node.
 */
final class StackNode<Type>(x:Type) {
  next:StackNode<Type>?;
  x:Type <- x;
}
