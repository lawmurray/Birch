/**
 * Record of a FactorEvent.
 *
 * - w: Log-weight.
 */
final class FactorRecord(w:Real) < Record {
  /**
   * Log-weight.
   */
  w:Real <- w;
}

/**
 * Create a FactorRecord.
 */
function FactorRecord(w:Real) -> FactorRecord {
  evt:FactorRecord(w);
  return evt;
}
