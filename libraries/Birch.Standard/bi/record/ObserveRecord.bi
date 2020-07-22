/**
 * Record of an ObserveEvent.
 *
 * - x: Observed value.
 */
final class ObserveRecord<Value>(x:Value) < Record {
  /**
   * Observed value.
   */
  x:Value <- x;
}

/**
 * Create an ObserveRecord.
 */
function ObserveRecord<Value>(x:Value) -> ObserveRecord<Value> {
  return construct<ObserveRecord<Value>>(x);
}
