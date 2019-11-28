/**
 * Event handler that applies the *undelay* action to each event.
 *
 * !!! tip
 *     UndelayHandler is thread-safe, and can be used via the singleton
 *     `undelay`.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
final class UndelayHandler < TraceHandler {
  function handle(record:Record, event:Event) -> Real {
    return event.undelay(record);
  }
}

/**
 * Singleton UndelayHandler.
 */
undelay:UndelayHandler;
