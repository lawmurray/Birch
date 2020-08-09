/**
 * Record of an event as entered into a trace.
 *
 * The Record class hierarchy is analogous the Event class hierarchy. It
 * exists because events may contain more information than is necessary to
 * record; Event objects are translated to Record objects before entering
 * into a trace.
 */
abstract class Record {
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
