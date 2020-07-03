/*
 * Tape node.
 */
final class TapeNode<Type>(x:Type) {
  next:TapeNode<Type>?;
  x:Type <- x;
}
