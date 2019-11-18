/**
 * Record of an immediate value.
 *
 * - v: The immediate value.
 */
final class ImmediateRecord<Value>(v:Value) < ValueRecord<Value> {
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
 * Create an ImmediateRecord.
 */
function ImmediateRecord<Value>(v:Value) -> ImmediateRecord<Value> {
  evt:ImmediateRecord<Value>(v);
  return evt;
}
