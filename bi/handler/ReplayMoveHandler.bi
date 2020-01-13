/**
 * Event handler that applies the *replayMove* action to each event.
 *
 * !!! tip
 *     ReplayMoveHandler is thread-safe, and can be used via the singleton
 *     `replayMove`.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
final class ReplayMoveHandler < TraceHandler {
  function handle(record:Record, event:Event) -> Real {
    return event.replayMove(record);
  }
}

/**
 * Singleton ReplayMoveHandler.
 */
replayMove:ReplayMoveHandler;
