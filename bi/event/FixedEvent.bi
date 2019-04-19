/**
 * Event to store a fixed value.
 *
 * - v: The fixed value.
 */
final class FixedEvent<Value>(v:Value) < ValueEvent<Value> {
  /**
   * Value associated with the event.
   */
  v:Value <- v;

  function hasValue() -> Boolean {
    return true;
  }

  function value() -> Value {
    return v;
  }

  function playImmediate() -> Real {
    return 0.0;
  }
}

/**
 * Create a FixedEvent.
 */
function FixedEvent<Value>(v:Value) -> FixedEvent<Value> {
  evt:FixedEvent<Value>(v);
  return evt;
}
