/**
 * Event handler that applies the *replayDelay* action to each event.
 *
 * !!! tip
 *     ReplayDelayHandler is thread-safe, and can be used via the singleton
 *     `replayDelay`.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
final class ReplayDelayHandler < TraceHandler {
  function handle(record:Record, event:Event) -> Real {
    return event.replayDelay(record);
  }
}

/**
 * Singleton ReplayDelayHandler.
 */
replayDelay:ReplayDelayHandler;
