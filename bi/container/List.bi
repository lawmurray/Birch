/**
 * List.
 */
class List<Type> {
  head:ListNode<Type>?;
  tail:ListNode<Type>&;
  count:Integer <- 0;

  /**
   * Get the size of the list.
   */
  function size() -> Integer {
    return count;
  }

  /**
   * Is this list empty?
   */
  function empty() -> Boolean {
    return count == 0;
  }

  /**
   * Clear the contents of the list.
   */
  function clear() {
    head <- nil;
    tail <- nil;
    count <- 0;
  }

  /**
   * Get the first element of the list.
   */
  function front() -> Type {
    assert head?;
    return head!.x;
  }

  /**
   * Get the last element of the list.
   */
  function back() -> Type {
    tail:ListNode<Type>? <- this.tail;
    assert tail?;
    return tail!.x;
  }

  /**
   * Insert a new element at the start of the list.
   *
   * - x: the element.
   */
  function pushFront(x:Type) {
    node:ListNode<Type>;
    node.x <- x;
    
    if (head?) {
      head!.prev <- node;
      node.next <- head;
    } else {
      tail <- node;
    }
    head <- node;
    count <- count + 1;
  }

  /**
   * Insert a new element at the end of the list.
   *
   * - x: the element.
   */
  function pushBack(x:Type) {
    tail:ListNode<Type>? <- this.tail;
    node:ListNode<Type>;
    node.x <- x;
    
    if (tail?) {
      tail!.next <- node;
      node.prev <- this.tail;
    } else {
      head <- node;
    }
    this.tail <- node;
    count <- count + 1;
  }

  /**
   * Remove the first element of the list.
   */
  function popFront() {
    assert head?;
    head <- head!.popFront();
    count <- count - 1;
    if (count <= 1) {
      tail <- head;
    }
  }

  /**
   * Remove the last element of the list
   */
  function popBack() {
    tail:ListNode<Type>? <- this.tail;
    assert tail?;
    this.tail <- tail!.popBack();
    count <- count - 1;
    if (count <= 1) {
      head <- tail;
    }
  }

  /**
   * Insert a new element into the list.
   *
   * - i: the position at which to insert,
   * - x: the element.
   *
   * Inserts the new element immediately before the current element at
   * position `i`. To insert at the end of the list, use a position that is
   * one more than the current size of the list, or `pushBack()`.
   */
  function insert(i:Integer, x:Type) {
    assert 1 <= i && i <= count + 1;
    if (i == 1) {
      pushFront(x);
    } else if (i == count + 1) {
      pushBack(x);
    } else {
      node:ListNode<Type>;
      node.x <- x;
      get(i).insert(node);
      count <- count + 1;
    }
  }

  /**
   * Erase an element of the list.
   *
   * - i: the position of the element.
   */
  function erase(i:Integer) {
    assert 1 <= i && i <= count;
    if (i == 1) {
      popFront();
    } else if (i == count) {
      popBack();
    } else {
      get(i).erase();
      count <- count - 1;
    }
  }

  /**
   * Iterate over the elements of the list.
   *
   * Return: a fiber object that yields each element in forward order.
   */
  fiber walk() -> Type! {
    node:ListNode<Type>? <- head;
    while (node?) {
      yield node!.x;
      node <- node!.next;
    }
  }

  /**
   * Get an arbitrary node of the list.
   */
  function get(i:Integer) -> ListNode<Type> {
    assert 1 <= i && i <= count;
    node:ListNode<Type>?;
    if (2*i <= count) {
      /* walk forward */
      node <- head;
      for (j:Integer in 1..i-1) {
        assert node?;
        node <- node!.next;
      }
    } else {
      /* walk backward */
      node <- tail;
      for (j:Integer in 1..count - i) {
        assert node?;
        node <- node!.prev;
      }
    }
    assert node?;
    return node!;
  }
}
