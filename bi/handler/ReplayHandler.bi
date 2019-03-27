/**
 * Abstract event handler that replays a trace of events.
 *
 * - replay: The trqce to replay.
 */
class ReplayHandler(replay:List<Event>) < Handler {
  /**
   * Trace of events to replay.
   */
  replay:List<Event> <- replay;
}
