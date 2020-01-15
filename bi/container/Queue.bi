/**
 * Double-ended queue.
 */
final class Queue<Type> < DoubleStack<Type> {
  /**
   * Get the first element.
   */
  function front() -> Type {
    if !forward? {
      allForward();
    }
    return topForward();
  }

  /**
   * Get the last element.
   */
  function back() -> Type {
    if !backward? {
      allBackward();
    }
    return topBackward();
  }

  /**
   * Insert a new element at the start.
   *
   * - x: Value.
   */
  function pushFront(x:Type) {
    pushForward(x);
  }

  /**
   * Insert a new element at the end.
   *
   * - x: Value.
   */
  function pushBack(x:Type) {
    pushBackward(x);
  }

  /**
   * Remove the first element and return it.
   */
  function popFront() -> Type {
    return popForward();
  }

  /**
   * Remove the last element and return it.
   */
  function popBack() -> Type {
    return popBackward();
  }

  /**
   * Forward iteration.
   *
   * Return: a fiber object that yields each element in forward order.
   */
  fiber walk() -> Type {
    allForward();
    while forward? {
      yield forward!.x;
      oneBackward();
    }
  }
  
  function read(buffer:Buffer) {
    auto f <- buffer.walk();
    while f? {
      /* tricky, but works for both value and class types */
      auto x <- make<Type>();
      auto y <- f!.get(x);
      if y? {
        x <- Type?(y);  // cast needed for y:Object?
        pushBack(x!);
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
