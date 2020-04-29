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
  function handle(events:Event!) -> Expression<Real>? {
    w:Expression<Real>?;
    while events? {
      auto v <- handle(events!);
      if v? {
        if w? {
          w <- w! + v!;
        } else {
          w <- v;
        }
      }
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
  function handle(events:Event!, output:Trace) -> Expression<Real>? {
    w:Expression<Real>?;
    while events? {
      auto v <- handle(events!);
      if v? {
        if w? {
          w <- w! + v!;
        } else {
          w <- v;
        }
      }
      output.pushBack(events!.record());
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
  abstract function handle(event:Event) -> Expression<Real>?;

  /**
   * Handle a sequence of events with an input trace.
   *
   * - input: Input trace.
   * - events: Event sequence.
   *
   * Returns: Accumulated log-weight.
   */
  function handle(input:Trace, events:Event!) -> Expression<Real>? {
    w:Expression<Real>?;
    while events? {
      auto v <- handle(input.here(), events!);
      if v? {
        if w? {
          w <- w! + v!;
        } else {
          w <- v;
        }
      }
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
  function handle(input:Trace, events:Event!, output:Trace) ->
      Expression<Real>? {
    w:Expression<Real>?;
    while events? {
      auto v <- handle(input.here(), events!);
      if v? {
        if w? {
          w <- w! + v!;
        } else {
          w <- v;
        }
      }
      input.next();
      output.pushBack(events!.record());
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
  abstract function handle(record:Record, event:Event) -> Expression<Real>?;
}
