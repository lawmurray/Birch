/**
 * Doubly-linked list. Beyond its typical uses, because List is a recursive
 * data structure, it provides reasonably good sharing under Birch's lazy
 * deep clone mechanism, although not as good as Stack and Queue, as they are
 * singly-linked.
 *
 * !!! caution
 *     Long lists (length in the tens of thousands) can cause a segfault on
 *     destruction due to stack overflow. Possible solutions are:
 *
 *     1. Use Vector instead.
 *     2. Remove items one-by-one before the List object goes out of scope
 *        (with e.g. `popBack()`).
 *     3. Increase the stack size with `ulimit` or similar.
 */
class List<Type> {
  head:ListNode<Type>?;
  tail:ListNode<Type>&;
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
    tail <- nil;
    count <- 0;
  }

  /**
   * Copy the list.
   */
  function copy() -> Queue<Type> {
    o:Queue<Type>;
    auto f <- walk();
    while (f?) {
      o.pushBack(f!);
    }
    return o;
  }

  /**
   * Get the first element.
   */
  function front() -> Type {
    assert head?;
    return head!.x;
  }

  /**
   * Get the last element.
   */
  function back() -> Type {
    tail:ListNode<Type>? <- this.tail;
    assert tail?;
    return tail!.x;
  }

  /**
   * Get an element.
   *
   * - i: Position.
   */
  function get(i:Integer) -> Type {
    return getNode(i).x;
  }

  /**
   * Set an element.
   *
   * - i: Position.
   * - x: Value.
   */
  function set(i:Integer, x:Type) {
    getNode(i).x <- x;
  }

  /**
   * Insert a new element at the start.
   *
   * - x: Value.
   */
  function pushFront(x:Type) {
    node:ListNode<Type>(x);
    if empty() {
      tail <- node;
    } else {
      assert head?;
      head!.prev <- node;
      node.next <- head;
    }
    head <- node;
    count <- count + 1;
    
    assert this.head? && this.tail?;
  }

  /**
   * Insert a new element at the end.
   *
   * - x: Value.
   */
  function pushBack(x:Type) {
    node:ListNode<Type>(x);
    if empty() {
      head <- node;
    } else {
      tail:ListNode<Type>? <- this.tail;
      assert tail?;
      tail!.next <- node;
      node.prev <- tail!;
    }
    tail <- node;
    count <- count + 1;
    
    assert this.head? && this.tail?;
  }

  /**
   * Remove the first element.
   */
  function popFront() {
    assert !empty();
    
    assert head?;
    head <- head!.popFront();
    count <- count - 1;
    if count <= 1 {
      tail <- head;
    }
    
    assert this.count == 0 || (this.head? && this.tail?);
    assert this.count > 0 || (!this.head? && !this.tail?);
  }

  /**
   * Remove the last element.
   */
  function popBack() {
    assert !empty();

    tail:ListNode<Type>? <- this.tail;
    assert tail?;
    this.tail <- tail!.popBack();
    count <- count - 1;
    if count <= 1 {
      head <- this.tail;
    }
    
    assert this.count == 0 || (this.head? && this.tail?);
    assert this.count > 0 || (!this.head? && !this.tail?);
  }
  
  /**
   * Insert a new element.
   *
   * - i: Position.
   * - x: Value.
   *
   * Inserts the new element immediately before the current element at
   * position `i`. To insert at the end of the container, use a position that
   * is one more than the current size, or `pushBack()`.
   */
  function insert(i:Integer, x:Type) {
    assert 1 <= i && i <= count + 1;
    
    if (i == 1) {
      pushFront(x);
    } else if (i == count + 1) {
      pushBack(x);
    } else {
      node:ListNode<Type>(x);
      getNode(i).insert(node);
      count <- count + 1;
    }
    
    assert this.head? && this.tail?;
  }

  /**
   * Erase an element.
   *
   * - i: Position.
   */
  function erase(i:Integer) {
    assert 1 <= i && i <= count;
    
    if (i == 1) {
      popFront();
    } else if (i == count) {
      popBack();
    } else {
      getNode(i).erase();
      count <- count - 1;
    }
    
    assert this.count == 0 || (this.head? && this.tail?);
    assert this.count > 0 || (!this.head? && !this.tail?);
  }

  /**
   * Forward iteration.
   *
   * Return: a fiber object that yields each element in forward order.
   */
  fiber walk() -> Type {
    node:ListNode<Type>? <- head;
    while (node?) {
      yield node!.x;
      node <- node!.next;
    }
  }

  /**
   * First node, if any. This can be used to maintain a bidirectional
   * iterator over the container.
   */
  function begin() -> ListNode<Type>? {
    return head;
  }
  
  /**
   * Last node, if any. This can be used to maintain a bidirectional
   * iterator over the container.
   */
  function end() -> ListNode<Type>? {
    return tail;
  }

  /**
   * Get a node.
   *
   * - i: Position.
   */
  function getNode(i:Integer) -> ListNode<Type> {
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
  
  function read(buffer:Buffer) {
    auto f <- buffer.walk();
    while f? {
      /* tricky, but works for both basic and class types */
      x:Type;
      auto y <- f!.get(x);
      if (y?) {
        x <- Type?(y)!;  // cast needed for y:Object?
        pushBack(x);
      }
    }
  }

  function write(buffer:Buffer) {
    buffer.setArray();
    auto f <- walk();
    while (f?) {
      buffer.push().set(f!);
    }
  }
}
