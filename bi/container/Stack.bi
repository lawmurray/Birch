/**
 * Stack.
 */
class Stack<Type> {
  head:StackNode<Type>?;
  count:Integer <- 0;

  /**
   * Number of elements.
   */
  function size() -> Integer {
    return count;
  }

  /**
   * Is this empty?
   */
  function empty() -> Boolean {
    return count == 0;
  }

  /**
   * Clear all elements.
   */
  function clear() {
    head <- nil;
    count <- 0;
  }

  /**
   * Get the top element.
   */
  function top() -> Type {
    assert head?;
    return head!.x;
  }

  /**
   * Push an element onto the top.
   *
   * - x: the element.
   */
  function push(x:Type) {
    node:StackNode<Type>(x);    
    node.next <- head;
    head <- node;
    count <- count + 1;
  }

  /**
   * Pop an element from the top.
   */
  function pop() {
    assert head?;
    head <- head!.next;
    count <- count - 1;
  }
}
