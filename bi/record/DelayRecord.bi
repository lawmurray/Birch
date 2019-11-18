/**
 * Record of a delayed value.
 *
 * - v: The random variate that will contain the delayed value.
 */
final class DelayRecord<Value>(v:Random<Value>) < ValueRecord<Value> {
  /**
   * Random variate that will contain the delayed value.
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
 * Create a DelayRecord.
 */
function DelayRecord<Value>(v:Random<Value>) -> DelayRecord<Value> {
  evt:DelayRecord<Value>(v);
  return evt;
}
