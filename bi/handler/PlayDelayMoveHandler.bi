/**
 * Event handler that applies the *playDelayMove* action to each event.
 *
 * !!! tip
 *     PlayDelayMoveHandler is thread-safe, and can be used via the singleton
 *     `playDelayMove`.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
final class PlayDelayMoveHandler < Handler {
  function handle(event:Event) -> Real {
    return event.playDelayMove();
  }
}

/**
 * Singleton PlayDelayMoveHandler.
 */
playDelayMove:PlayDelayMoveHandler;
