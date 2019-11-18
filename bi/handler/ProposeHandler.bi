/**
 * Event handler that applies the *propoose* action to each event.
 *
 * - trace: The trace.
 */
final class ProposeHandler(trace:Trace) < TraceHandler(trace) {
  function handle(event:Event, record:Record) -> Real {
    return event.propose(record);
  }
}
