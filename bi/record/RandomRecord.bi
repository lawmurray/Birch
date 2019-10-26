/**
 * Record of a random value.
 *
 * - v: The random value.
 */
final class RandomRecord<Value>(v:Random<Value>) < ValueRecord<Value> {
  /**
   * Value associated with the record.
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
 * Create a RandomRecord.
 */
function RandomRecord<Value>(v:Random<Value>) -> RandomRecord<Value> {
  evt:RandomRecord<Value>(v);
  return evt;
}
