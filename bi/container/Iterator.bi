/**
 * Stateful list, used as an iterator. An Iterator maintains a current
 * position in the list, and offers $O(1)$ insertions and deletions at that
 * position, but $O(N)$ insertion and deletion at any other position,
 * including at the front and back.
 */
final class Iterator<Type> < DoubleStack<Type> {
  /**
   * Is there an element at the current position?
   */
  function hasHere() -> Boolean {
    return forward?;
  }

  /**
   * Is there an element before the current position?
   */
  function hasBefore() -> Boolean {
    return backward?;
  }

  /**
   * Get the first element.
   */
  function front() -> Type {
    allForward();
    return topForward();
  }

  /**
   * Get the last element.
   */
  function back() -> Type {
    allBackward();
    return topBackward();
  }

  /**
   * Get the element at the current position.
   */
  function here() -> Type {
    return topForward();
  }

  /**
   * Get the element before the current position.
   */
  function before() -> Type {
    return topBackward();
  }

  /**
   * Insert a new element at the start.
   *
   * - x: Value.
   */
  function pushFront(x:Type) {
    allForward();
    pushForward(x);
  }

  /**
   * Insert a new element at the end.
   *
   * - x: Value.
   */
  function pushBack(x:Type) {
    allBackward();
    pushBackward(x);
  }

  /**
   * Insert a new element at the current position.
   *
   * - x: Value.
   */
  function pushHere(x:Type) {
    pushForward(x);
  }

  /**
   * Insert a new element before the current position.
   *
   * - x: Value.
   */
  function pushBefore(x:Type) {
    pushBackward(x);
  }

  /**
   * Remove the first element and return it.
   */
  function popFront() -> Type {
    allForward();
    return popForward();
  }

  /**
   * Remove the last element and return it.
   */
  function popBack() -> Type {
    allBackward();
    return popBackward();
  }

  /**
   * Remove the element at the current position and return it.
   */
  function popHere() -> Type {
    return popForward();
  }

  /**
   * Remove the element before the current position and return it.
   */
  function popBefore() -> Type {
    return popBackward();
  }

  /**
   * Forward iteration.
   *
   * Return: a fiber object that yields each element in forward order.
   */
  fiber walk() -> Type {
    rewind();
    while hasHere() {
      yield here();
      next();
    }
  }
  
  /**
   * Move the current position forward one.
   */
  function next() {
    oneBackward();
  }
  
  /**
   * Move the current position backward one.
   */
  function previous() {
    oneForward();
  }
  
  /**
   * Rewind the current position back to the start.
   */
  function rewind() {
    allForward();
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
    rewind();
  }

  function write(buffer:Buffer) {
    buffer.setArray();
    auto f <- walk();
    while f? {
      buffer.push().set(f!);
    }
  }
}
