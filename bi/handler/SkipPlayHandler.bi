/**
 * Event handler that applies the *skip and play* action to each event. This
 * is used to skip records in a trace while otherwise applying the *play*
 * action.
 *
 * !!! tip
 *     SkipPlayHandler is thread-safe, and can be used via the singleton
 *     `replay`.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
final class SkipPlayHandler < TraceHandler {
  function handle(record:Record, event:Event) -> Real {
    return event.play();
  }
}

/**
 * Singleton SkipPlayHandler.
 */
skipPlay:SkipPlayHandler;
