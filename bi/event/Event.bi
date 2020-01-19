/**
 * Event during the simulation of a model.
 *
 * An Event requires that an action be taken. The choice of action is made
 * by a Handler. The action is performed by calling the appropriate
 * member function of event, such as `play()`, `playDelay()`, `playMove()`,
 * etc.
 */
abstract class Event {
  /**
   * Perform the *play* action.
   *
   * Returns: The required weight adjustment.
   */
  abstract function play() -> Real;

  /**
   * Perform the *playMove* action.
   *
   * Returns: The required weight adjustment.
   */
  abstract function playMove() -> Real;

  /**
   * Perform the *playDelay* action.
   *
   * Returns: The required weight adjustment.
   */
  abstract function playDelay() -> Real;

  /**
   * Perform the *playDelayMove* action.
   *
   * Returns: The required weight adjustment.
   */
  abstract function playDelayMove() -> Real;

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
   * Perform the *replayMove* action. This is typically used when
   * proposing a new trace that is a modication of a trace that was simulated
   * forward using `playMove()`.
   *
   * - record: Associated record in the trace.
   *
   * Returns: The required weight adjustment.
   */
  abstract function replayMove(record:Record) -> Real;

  /**
   * Perform the *replayDelay* action. This is typically used when
   * reconstructing a trace that was originally simulated forward using
   * `playDelay()`.
   *
   * - record: Associated record in the trace.
   *
   * Returns: The required weight adjustment.
   */
  abstract function replayDelay(record:Record) -> Real;

  /**
   * Perform the *replayDelayMove* action. This is typically used when
   * proposing a new trace that is a modication of a trace that was simulated
   * forward using `playDelayMove()`.
   *
   * - record: Associated record in the trace.
   *
   * Returns: The required weight adjustment.
   */
  abstract function replayDelayMove(record:Record) -> Real;

  /**
   * Produce a record for the event.
   *
   * Returns: The record.
   */
  abstract function record() -> Record;
}
