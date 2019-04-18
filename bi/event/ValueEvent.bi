/**
 * Abstract dummy event with a value.
 */
class ValueEvent<Value> < Event {
  function isValue() -> Boolean {
    return true;
  }

  /**
   * Is there a value associated with this event?
   */
  function hasValue() -> Boolean {
    assert false;
  }

  /**
   * Get the value.
   */
  function value() -> Value {
    assert false;
  }
}
