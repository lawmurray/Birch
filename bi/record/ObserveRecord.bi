/**
 * Record of an ObserveEvent.
 *
 * - v: Observed value.
 */
final class ObserveRecord<Value>(v:Value) < Record {
  /**
   * Observed value.
   */
  x:Value <- x;
}

/**
 * Create an ObserveRecord.
 */
function ObserveRecord<Value>(v:Value) -> ObserveRecord<Value> {
  evt:ObserveRecord<Value>(v);
  return evt;
}
