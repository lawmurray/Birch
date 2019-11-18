/**
 * Abstract event handler that requires an input trace.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
abstract class TraceHandler {
  /**
   * Handle a sequence of events with an input trace.
   *
   * - input: Input trace.
   * - events: Event sequence.
   *
   * Returns: Accumulated log-weight.
   */
  final function handle(input:Trace, events:Event!) -> Real {
    auto w <- 0.0;
    while w > -inf && events? {
      w <- w + handle(input.popFront(), events!);
    }
    return w;
  }

  /**
   * Handle a sequence of events with an input trace and record them in an
   * output trace.
   *
   * - input: Input trace.
   * - events: Event sequence.
   * - output: Output trace.
   *
   * Returns: Accumulated log-weight.
   */
  final function handle(input:Trace, events:Event!, output:Trace) -> Real {
    auto w <- 0.0;
    while w > -inf && events? {
      auto event <- events!;
      w <- w + handle(input.popFront(), event);
      output.pushBack(event.record());
    }
    return w;
  }
  
  /**
   * Handle an event with an input record.
   *
   * - record: The record.
   * - event: The event.
   *
   * Returns: Log-weight.
   */
  abstract function handle(record:Record, event:Event) -> Real;
}
