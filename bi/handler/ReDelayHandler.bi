/**
 * Event handler that applies the *redelay* action to each event.
 *
 * !!! tip
 *     ReDelayHandler is thread-safe, and can be used via the singleton
 *     `redelay`.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
final class ReDelayHandler < TraceHandler {
  function handle(record:Record, event:Event) -> Real {
    return event.redelay(record);
  }
}

/**
 * Singleton ReDelayHandler.
 */
redelay:ReDelayHandler;
