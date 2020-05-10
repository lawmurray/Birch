/**
 * Trace of a model execution. Each record in the trace corresponds to an
 * event emitted during execution.
 */
class Trace = Tape<Record>;

/**
 * Event handler that eagerly computes weights.
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
   * - Container: Type supporting `pushBack(Record)`.
   *
   * - events: Event sequence.
   * - trace: Output trace.
   *
   * Returns: Accumulated log-weight.
   */
  final function handle<Container>(events:Event!, output:Container) -> Real {
    auto w <- 0.0;
    while w > -inf && events? {
      w <- w + handle(events!);
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
   * - Container: Type supporting `pushBack(Record)`.
   *
   * - input: Input trace.
   * - events: Event sequence.
   * - output: Output trace.
   *
   * Returns: Accumulated log-weight.
   */
  final function handle<Container>(input:Trace, events:Event!,
      output:Container) -> Real {
    auto w <- 0.0;
    while w > -inf && events? {
      w <- w + handle(input.here(), events!);
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
  abstract function handle(record:Record, event:Event) -> Real;
}

/**
 * Handle a sequence of events using the default event handler. This is a
 * convenience function equivalent to `PlayHandler(true).handle(events)`.
 *
 * - events: Event sequence.
 *
 * Returns: Accumulated log-weight.
 */
function handle(events:Event!) -> Real {
  return PlayHandler(true).handle(events);
}
