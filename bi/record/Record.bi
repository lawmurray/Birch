/**
 * Abstract record in a trace.
 */
abstract class Record {
  /**
   * Does this have a value?
   */
  abstract function hasValue() -> Boolean;
  
  /**
   * Compute contribution to an acceptance ratio between this record (the
   * proposal) and another record (the current).
   *
   * - record: Another record.
   * - scale: Scale of the move.
   */
  function ratio(record:Record, scale:Real) -> Real {
    return 0.0;
  }
}
