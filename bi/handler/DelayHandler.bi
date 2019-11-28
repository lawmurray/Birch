/**
 * Event handler that applies the *delay* action to each event.
 *
 * !!! tip
 *     DelayHandler is thread-safe, and can be used via the singleton `delay`.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
final class DelayHandler < Handler {
  function handle(event:Event) -> Real {
    return event.delay();
  }
}

/**
 * Singleton DelayHandler.
 */
delay:DelayHandler;
