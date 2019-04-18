/**
 * Dummy event to store a fixed value, used for efficiently recording and
 * replaying traces without holding onto extraneous objects.
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
}

/**
 * Create a FixedEvent.
 */
function FixedEvent<Value>(v:Value) -> FixedEvent<Value> {
  evt:FixedEvent<Value>(v);
  return evt;
}
