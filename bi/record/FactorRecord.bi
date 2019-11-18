/**
 * Record of a factor.
 *
 * - w: Associated weight.
 */
final class FactorRecord(w:Real) < Record {
  /**
   * Associated weight.
   */
  w:Real <- w;

  function hasValue() -> Boolean {
    return false;
  }
}

/**
 * Create a FactorRecord.
 */
function FactorRecord(w:Real) -> FactorRecord {
  evt:FactorRecord(w);
  return evt;
}
