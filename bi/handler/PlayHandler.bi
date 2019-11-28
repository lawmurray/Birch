/**
 * Event handler that applies the *play* action to each event.
 *
 * !!! tip
 *     PlayHandler is thread-safe, and can be used via the singleton `play`.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
class PlayHandler < Handler {
  function handle(event:Event) -> Real {
    return event.play();
  }
}

/**
 * Singleton PlayHandler.
 */
play:PlayHandler;
