/**
 * Record of an AssumeEvent.
 *
 * - x: Random variate.
 */
final class AssumeRecord<Value>(x:Random<Value>) < Record {
  /**
   * Random variate.
   */
  x:Random<Value> <- x;
}

/**
 * Create a AssumeRecord.
 */
function AssumeRecord<Value>(v:Random<Value>) -> AssumeRecord<Value> {
  return construct<AssumeRecord<Value>>(v);
}
