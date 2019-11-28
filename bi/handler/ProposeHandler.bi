/**
 * Event handler that applies the *propoose* action to each event.
 *
 * !!! tip
 *     ProposeHandler is thread-safe, and can be used via the singleton
 *     `propose`.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
final class ProposeHandler < TraceHandler {
  function handle(record:Record, event:Event) -> Real {
    return event.propose(record);
  }
}

/**
 * Singleton ProposeHandler.
 */
propose:ProposeHandler;
