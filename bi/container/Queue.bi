/**
 * First in, first-out (FIFO) queue. Beyond its typical uses, because Queue is
 * a recursive data structure, it provides particularly good sharing under
 * Birch's lazy deep clone mechanism.
 *
 * !!! caution
 *     See note under List for possible segfault issues on the destruction
 *     of large queues.
 */
final class Queue<Type> {
  forward:QueueNode<Type>?;
  backward:QueueNode<Type>?;
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
    forward <- nil;
    backward <- nil;
    count <- 0;
  }

  /**
   * Get the first element.
   */
  function front() -> Type {
    assert !empty();
    if !forward? {
      allForward();
    }
    return forward!.x;
  }

  /**
   * Get the last element.
   */
  function back() -> Type {
    assert !empty();
    if !backward? {
      allBackward();
    }
    return backward!.x;
  }

  /**
   * Insert a new element at the start.
   *
   * - x: Value.
   */
  function pushFront(x:Type) {
    node:QueueNode<Type>(x);
    //node.next <- forward;
    if forward? {
      cpp{{
      node->next = std::move(forward.get());
      }}
    }
    forward <- node;
    count <- count + 1;
  }

  /**
   * Insert a new element at the end.
   *
   * - x: Value.
   */
  function pushBack(x:Type) {
    node:QueueNode<Type>(x);
    //node.next <- backward;
    if backward? {
      cpp{{
      node->next = std::move(backward.get());
      }}
    }
    backward <- node;
    count <- count + 1;
  }

  /**
   * Remove the first element.
   */
  function popFront() {
    assert !empty();
    if !forward? {
      allForward();
    }
    //forward <- forward!.next;
    if forward? {
      cpp{{
      forward = std::move(forward.get()->next);
      }}
    }
    count <- count - 1;
  }

  /**
   * Remove the last element.
   */
  function popBack() {
    assert !empty();
    if !backward? {
      allBackward();
    }
    //backward <- backward!.next;
    if backward? {
      cpp{{
      backward = std::move(backward.get()->next);
      }}
    }
    count <- count - 1;
  }

  /**
   * Forward iteration.
   *
   * Return: a fiber object that yields each element in forward order.
   */
  fiber walk() -> Type {
    allForward();
    auto node <- forward;
    while node? {
      yield node!.x;
      node <- node!.next;
    }
  }

  /**
   * First node, if any. This can be used to maintain a forward
   * iterator over the container.
   */
  function begin() -> QueueNode<Type>? {
    allForward();
    return forward;
  }
  
  /**
   * Last node, if any. This can be used to maintain a backward
   * iterator over the container.
   */
  function end() -> QueueNode<Type>? {
    allBackward();
    return backward;
  }
  
  /**
   * Move all elements to the forward list.
   */
  function allForward() {
    while backward? {
      cpp{{
      auto node = std::move(backward.get());
      backward = std::move(node->next);
      node->next = std::move(forward);
      forward = std::move(node);
      }}
    }
  }
  
  /**
   * Move all elements to the backward list.
   */
  function allBackward() {
    while forward? {
      cpp{{
      auto node = std::move(forward.get());
      forward = std::move(node->next);
      node->next = std::move(backward);
      backward = std::move(node);
      }}
    }
  }

  function read(buffer:Buffer) {
    auto f <- buffer.walk();
    while f? {
      /* tricky, but works for both basic and final class types */
      x:Type;
      auto y <- f!.get(x);
      if y? {
        x <- Type?(y)!;  // cast needed for y:Object?
        pushBack(x);
      }
    }
    allForward();
  }

  function write(buffer:Buffer) {
    buffer.setArray();
    auto f <- walk();
    while f? {
      buffer.push().set(f!);
    }
  }
}
