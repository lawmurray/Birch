/**
 * Event handler that applies the *skip and delay* action to each event.
 * This is used to skip records in a trace while otherwise applying the
 * *delay* action.
 *
 * !!! tip
 *     SkipDelayHandler is thread-safe, and can be used via the singleton
 *     `redelay`.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
final class SkipDelayHandler < TraceHandler {
  function handle(record:Record, event:Event) -> Real {
    return event.delay();
  }
}

/**
 * Singleton SkipDelayHandler.
 */
skipDelay:SkipDelayHandler;
