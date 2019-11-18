/**
 * Event handler.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
abstract class Handler {  
  /**
   * Handle a sequence of events.
   *
   * - events: Fiber yielding events.
   *
   * Returns: Log-weight.
   */
  final function handle(events:Event!) -> Real {
    auto w <- 0.0;
    while w > -inf && events? {
      w <- w + handle(events!);
    }
    return w;
  }

  /**
   * Handle an event.
   *
   * - event: The event.
   *
   * Returns: Log-weight adjustment.
   */  
  abstract function handle(event:Event) -> Real;
}
