/**
 * Abstract event emitted during the execution of a model.
 *
 * The Event class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Event.svg"></object>
 * </center>
 */
abstract class Event {
  /**
   * Accept an event handler.
   */
  abstract function accept(handler:PlayHandler) -> Real;
  
  /**
   * Accept an event handler.
   */
  abstract function accept(handler:MoveHandler) -> Expression<Real>?;

  /**
   * Accept an event handler.
   */
  abstract function accept(record:Record, handler:PlayHandler) -> Real;

  /**
   * Accept an event handler.
   */
  abstract function accept(record:Record, handler:MoveHandler) ->
      Expression<Real>?;

  /**
   * Make a record for the event, in order to enter into a trace.
   *
   * Returns: The record.
   */
  abstract function record() -> Record;
}
