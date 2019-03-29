/**
 * Event handler that records a trace of events.
 */
class TraceHandler < ReplayHandler {
  /**
   * Recorded trace of events.
   */
  record:List<Event>;
  
  /**
   * Pause recording of events?
   */
  pause:Boolean <- false;
  
  function handle(evt:FactorEvent) -> Real {
    if !pause {
      record.pushBack(evt);
    }
    return super.handle(evt);
  }
  
  function handle(evt:RandomEvent) -> Real {
    if !pause {
      record.pushBack(evt);
    }
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

  /**
   * Set the pause flag. When true, events are not recorded.
   */
  function setPause(pause:Boolean) {
    this.pause <- pause;
  }
}
