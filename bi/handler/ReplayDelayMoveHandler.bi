/**
 * Event handler that applies the *replayDelayMove* action to each event.
 *
 * !!! tip
 *     ReplayDelayMoveHandler is thread-safe, and can be used via the
 *     singleton `replayDelayMove`.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
final class ReplayDelayMoveHandler < TraceHandler {
  function handle(record:Record, event:Event) -> Real {
    return event.replayDelayMove(record);
  }
}

/**
 * Singleton ReplayDelayMoveHandler.
 */
replayDelayMove:ReplayDelayMoveHandler;
