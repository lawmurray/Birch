/**
 * Event handler that records a trace of events.
 */
class TraceHandler < ReplayHandler {
  /**
   * Recorded trace of events.
   */
  record:List<Event>;
  
  function handle(evt:FactorEvent) -> Real {
    record.pushBack(evt);
    return super.handle(evt);
  }
  
  function handle(evt:RandomEvent) -> Real {
    record.pushBack(evt);
    return super.handle(evt);
  }

  /**
   * Remove and return the recorded trace.
   */
  function takeRecord() -> List<Event> {
    empty:List<Event>;
    auto record <- this.record;
    this.record <- empty;
    return record;
  }
  
  /**
   * Set the recorded trace.
   */
  function setRecord(record:List<Event>) {
    this.record <- record;
  }
}
