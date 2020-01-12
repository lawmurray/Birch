/**
 * Event handler that applies the *replay* action to each event.
 *
 * !!! tip
 *     RePlayHandler is thread-safe, and can be used via the singleton
 *     `replay`.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
final class RePlayHandler < TraceHandler {
  function handle(record:Record, event:Event) -> Real {
    return event.replay(record);
  }
}

/**
 * Singleton RePlayHandler.
 */
replay:RePlayHandler;
