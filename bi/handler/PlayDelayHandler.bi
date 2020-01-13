/**
 * Event handler that applies the *playDelay* action to each event.
 *
 * !!! tip
 *     PlayDelayHandler is thread-safe, and can be used via the singleton
 *     `playDelay`.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
final class PlayDelayHandler < Handler {
  function handle(event:Event) -> Real {
    return event.playDelay();
  }
}

/**
 * Singleton PlayDelayHandler.
 */
playDelay:PlayDelayHandler;
