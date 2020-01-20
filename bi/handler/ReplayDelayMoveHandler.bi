/**
 * Event handler that applies the *replayDelayMove* action to each event.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
final class ReplayDelayMoveHandler < TraceHandler {
  /**
   * Scale of the move.
   */
  scale:Real <- 0.1;

  function handle(record:Record, event:Event) -> Real {
    return event.replayDelayMove(record, scale);
  }
}

/**
 * Create a ReplayDelayMoveHandler.
 *
 * - scale: Scale of the move.
 */
function ReplayDelayMoveHandler(scale:Real) -> ReplayDelayMoveHandler {
  o:ReplayDelayMoveHandler;
  o.scale <- scale;
  return o;
}
