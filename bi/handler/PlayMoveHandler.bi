/**
 * Event handler that applies the *playMove* action to each event.
 *
 * !!! tip
 *     PlayMoveHandler is thread-safe, and can be used via the singleton
 *     `playMove`.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
final class PlayMoveHandler < Handler {
  function handle(event:Event) -> Real {
    return event.playMove();
  }
}

/**
 * Singleton PlayMoveHandler.
 */
playMove:PlayMoveHandler;
