/**
 * Event handler that applies the *redelay* action to each event.
 */
final class RedelayHandler < TraceHandler {
  function handle(event:Event, record:Record) -> Real {
    return event.redelay(record);
  }
}
