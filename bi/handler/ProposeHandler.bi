/**
 * Event handler that applies the *propoose* action to each event.
 */
final class ProposeHandler < TraceHandler {
  function handle(event:AssumeEvent, record:Record) -> Real {
    return event.propose(record);
  }
}
