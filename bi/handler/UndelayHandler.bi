/**
 * Event handler that applies the *undelay* action to each event.
 *
 * - trace: The trace.
 */
final class UndelayHandler(trace:Trace) < TraceHandler(trace) {
  function handle(event:Event, record:Record) -> Real {
    return event.undelay(record);
  }
}

/**
 * Create an UndelayHandler.
 */
function UndelayHandler(trace:Trace) -> UndelayHandler {
  o:UndelayHandler(trace);
  return o;
}
