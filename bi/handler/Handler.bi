/**
 * Abstract event handler.
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
   * - events: Event sequence.
   *
   * Returns: Accumulated (log-)weight.
   */
  final function handle(events:Event!) -> Real {
    auto w <- 0.0;
    while w > -inf && events? {
      w <- w + handle(events!);
    }
    return w;
  }

  /**
   * Handle a sequence of events and record them in an output trace.
   *
   * - events: Event sequence.
   * - trace: Output trace.
   *
   * Returns: Accumulated (log-)weight.
   */
  final function handle(events:Event!, output:Trace) -> Real {
    auto w <- 0.0;
    while w > -inf && events? {
      auto event <- events!;
      w <- w + handle(event);
      output.pushBack(event.record());
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
