/**
 * Event handler that applies the *redelay* action to each event.
 */
final class RedelayHandler(trace:Trace) < TraceHandler(trace) {
  function handle(event:Event, record:Record) -> Real {
    return event.redelay(record);
  }
}

/**
 * Create a RedelayHandler.
 */
function RedelayHandler(trace:Trace) -> RedelayHandler {
  o:RedelayHandler(trace);
  return o;
}
