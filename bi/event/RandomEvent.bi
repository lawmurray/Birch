/**
 * Dummy event to store a random value, used for efficiently recording and
 * replaying traces without holding onto extraneous objects.
 *
 * - v: The random value.
 */
final class RandomEvent<Value>(v:Random<Value>) < ValueEvent<Value> {
  /**
   * Random associated with the event.
   */
  v:Random<Value> <- v;

  function hasValue() -> Boolean {
    return v.hasValue();
  }

  function value() -> Value {
    return v.value();
  }
}

/**
 * Create a RandomEvent.
 */
function RandomEvent<Value>(v:Random<Value>) -> RandomEvent<Value> {
  evt:RandomEvent<Value>(v);
  return evt;
}
