/**
 * Resizeable vector with $O(1)$ random access.
 */
final class Vector<Type> {
  /**
   * Elements.
   */
  values:Type[_];

  /**
   * Number of elements.
   */
  function size() -> Integer {
    return length(values);
  }

  /**
   * Is this empty?
   */
  function empty() -> Boolean {
    return size() == 0;
  }

  /**
   * Clear all elements.
   */
  function clear() {
    values:Type[_];
    this.values <- values;
  }
  
  /**
   * Get an element.
   *
   * - i: Position.
   */
  function get(i:Integer) -> Type {
    return values[i];
  }

  /**
   * Set an element.
   *
   * - i: Position.
   * - x: Value.
   */
  function set(i:Integer, x:Type) {
    values[i] <- x;
  }

  /**
   * Get the first element.
   */
  function front() -> Type {
    assert size() > 0;
    return values[1];
  }

  /**
   * Get the last element.
   */
  function back() -> Type {
    assert size() > 0;
    return values[size()];
  }

  /**
   * Insert a new element at the start.
   *
   * - x: Value.
   */
  function pushFront(x:Type) {
    insert(1, x);
  }

  /**
   * Insert a new element at the end.
   *
   * - x: Value.
   */
  function pushBack(x:Type) {
    insert(size() + 1, x);
  }

  /**
   * Remove the first element.
   */
  function popFront() {
    erase(1);
  }

  /**
   * Remove the last element.
   */
  function popBack() {
    erase(size());
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
    assert 1 <= i && i <= size() + 1;
    cpp{{
    this_()->values.insert(i - 1, x);
    }}
  }

  /**
   * Erase an element.
   *
   * - i: Position.
   *
   * The size decreases by one.
   */
  function erase(i:Integer) {
    assert 1 <= i && i <= size();
    cpp{{
    this_()->values.erase(i - 1);
    }}
  }

  /**
   * Erase multiple elements.
   *
   * - i: Position.
   * - n: Number of elements.
   *
   * The size decreases by `n`.
   */
  function erase(i:Integer, n:Integer) {
    assert 1 <= i && i <= size();
    assert 1 <= n && n <= size() - i + 1;
    cpp{{
    this_()->values.erase(i - 1, n);
    }}
  }

  /**
   * Iterate over the elements.
   *
   * Return: a fiber object that yields each element in forward order.
   */
  fiber walk() -> Type {
    for i in 1..size() {
      yield values[i];
    }
  }

  /**
   * Convert to array.
   */
  function toArray() -> Type[_] {
    return values;
  }
  
  /**
   * Convert from array.
   */
  function fromArray(x:Type[_]) {
    values <- x;
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
  }

  function write(buffer:Buffer) {
    buffer.setArray();
    auto f <- walk();
    while f? {
      buffer.push().set(f!);
    }
  }
}
