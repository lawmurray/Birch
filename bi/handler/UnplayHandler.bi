/**
 * Event handler that applies the *unplay* action to each event.
 *
 * !!! tip
 *     UnplayHandler is thread-safe, and can be used via the singleton
 *     `unplay`.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
final class UnplayHandler < TraceHandler {
  function handle(record:Record, event:Event) -> Real {
    return event.unplay(record);
  }
}

/**
 * Singleton UnplayHandler.
 */
unplay:UnplayHandler;
