/**
 * Record of a SimulateEvent.
 *
 * - x: Simulated value.
 */
final class SimulateRecord<Value>(x:Value) < Record {
  /**
   * Simulated value.
   */
  x:Value <- x;
}

/**
 * Create a SimulateRecord.
 */
function SimulateRecord<Value>(v:Value) -> SimulateRecord<Value> {
  evt:SimulateRecord<Value>(v);
  return evt;
}
