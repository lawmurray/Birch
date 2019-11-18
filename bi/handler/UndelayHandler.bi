/**
 * Event handler that applies the *undelay* action to each event.
 */
final class UndelayHandler < TraceHandler {
  function handle(event:Event, record:Record) -> Real {
    return event.undelay(record);
  }
}
