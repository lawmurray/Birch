/**
 * Event during the simulation of a model.
 *
 * An Event requires that an action be taken. The choice of action is made
 * by a Handler. The action is performed by calling the appropriate
 * member function of event, such as `play()`, `delay()`, `replay()`, etc.
 */
abstract class Event {
  /**
   * Perform the *play* action.
   *
   * Returns: The required weight adjustment.
   */
  abstract function play() -> Real;

  /**
   * Perform the *replay* action. This is typically used when reconstructing
   * a trace that was originally simulated forward using `play()`.
   *
   * - record: Associated record in the trace.
   *
   * Returns: The required weight adjustment.
   */
  abstract function replay(record:Record) -> Real;

  /**
   * Perform the *delay* action.
   *
   * Returns: The required weight adjustment.
   */
  abstract function delay() -> Real;

  /**
   * Perform the *redelay* action. This is typically used when reconstructing
   * a trace that was originally simulated forward using `delay()`.
   *
   * - record: Associated record in the trace.
   *
   * Returns: The required weight adjustment.
   */
  abstract function redelay(record:Record) -> Real;

  /**
   * Perform the *propose* action.
   *
   * - record: Associated record in the trace.
   *
   * Returns: The required weight adjustment.
   */
  abstract function propose(record:Record) -> Real;
  
  /**
   * Produce a record for the event.
   *
   * Returns: The record.
   */
  abstract function record() -> Record;
}
