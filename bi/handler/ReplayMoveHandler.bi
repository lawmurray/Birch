/**
 * Event handler that applies the *replayMove* action to each event.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
final class ReplayMoveHandler < TraceHandler {
  /**
   * Scale of the move.
   */
  scale:Real <- 0.1;

  function handle(record:Record, event:Event) -> Real {
    return event.replayMove(record, scale);
  }
}


/**
 * Create a ReplayMoveHandler.
 *
 * - scale: Scale of the move.
 */
function ReplayMoveHandler(scale:Real) -> ReplayMoveHandler {
  o:ReplayMoveHandler;
  o.scale <- scale;
  return o;
}
