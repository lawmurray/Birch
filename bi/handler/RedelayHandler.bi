/**
 * Event handler that applies the *redelay* action to each event.
 *
 * !!! tip
 *     RedelayHandler is thread-safe, and can be used via the singleton
 *     `redelay`.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
final class RedelayHandler < TraceHandler {
  function handle(record:Record, event:Event) -> Real {
    return event.redelay(record);
  }
}

/**
 * Singleton RedelayHandler.
 */
redelay:RedelayHandler;
