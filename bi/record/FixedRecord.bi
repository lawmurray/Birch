/**
 * Record of a fixed value.
 *
 * - v: The fixed value.
 */
final class FixedRecord<Value>(v:Value) < ValueRecord<Value> {
  /**
   * Value associated with the record.
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
 * Create a FixedRecord.
 */
function FixedRecord<Value>(v:Value) -> FixedRecord<Value> {
  evt:FixedRecord<Value>(v);
  return evt;
}
