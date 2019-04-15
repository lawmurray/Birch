/*
 * List node.
 */
final class ListNode<Type>(x:Type) {
  x:Type <- x;
  prev:ListNode<Type>&;
  next:ListNode<Type>?;

  /**
   * Get value.
   */
  function getValue() -> Type {
    return x;
  }
  
  /**
   * Get previous node.
   */
  function getPrevious() -> ListNode<Type>? {
    return prev;
  }
  
  /**
   * Get next node.
   */
  function getNext() -> ListNode<Type>? {
    return next;
  }

  /**
   * Insert a new node just before this one in the list.
   *
   * - node: the new node.
   */
  function insert(node:ListNode<Type>) {
    prev:ListNode<Type>? <- this.prev;
    assert prev?;

    prev!.next <- node;
    node.prev <- prev;
    node.next <- this;
    this.prev <- node;
  }
  
  /**
   * Remove this node from the list.
   */
  function erase() {
    prev:ListNode<Type>? <- this.prev;
    assert prev?;
    assert next?;
    
    prev!.next <- next;
    next!.prev <- prev;
    this.next <- nil;
    this.prev <- nil;
  }
  
  /**
   * Pop this node from the front of the list.
   *
   * Return: the new front node of the list.
   */
  function popFront() -> ListNode<Type>? {
    node:ListNode<Type>? <- this.next;
    if node? {
      node!.prev <- nil;
    }
    this.next <- nil;
    this.prev <- nil;
    return node;
  }

  /**
   * Pop this node from the back of the list.
   *
   * Return: the new back node of the list.
   */
  function popBack() -> ListNode<Type>? {
    node:ListNode<Type>? <- this.prev;
    if node? {
      node!.next <- nil;
    }
    this.next <- nil;
    this.prev <- nil;
    return node;
  }
}
