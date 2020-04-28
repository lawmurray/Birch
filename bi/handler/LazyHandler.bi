/**
 * Abstract event handler that lazily computes weights.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
abstract class LazyHandler {
  /**
   * Handle a sequence of events.
   *
   * - events: Event sequence.
   *
   * Returns: Accumulated log-weight.
   */
  function handle(events:Event!) -> Real {
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
   * Returns: Accumulated log-weight.
   */
  function handle(events:Event!, output:Trace) -> Real {
    auto w <- 0.0;
    while w > -inf && events? {
      auto event <- events!;
      w <- w + handle(event);
      output.pushBack(event.record());
    }
    return w;
  }

  /**
   * Handle a single event.
   *
   * - event: The event.
   *
   * Returns: Log-weight.
   */
  abstract function handle(event:Event) -> Real;

  /**
   * Handle a sequence of events with an input trace.
   *
   * - input: Input trace.
   * - events: Event sequence.
   *
   * Returns: Accumulated log-weight.
   */
  function handle(input:Trace, events:Event!) -> Real {
    auto w <- 0.0;
    while w > -inf && events? {
      w <- w + handle(input.here(), events!);
      input.next();
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
  function handle(input:Trace, events:Event!, output:Trace) -> Real {
    auto w <- 0.0;
    while w > -inf && events? {
      auto event <- events!;
      w <- w + handle(input.here(), event);
      input.next();
      output.pushBack(event.record());
    }
    return w;
  }

  /**
   * Handle a single event with an input record.
   *
   * - record: The input record.
   * - event: The event.
   *
   * Returns: Log-weight.
   */
  abstract function handle(record:Record, event:Event) -> Real;
}
