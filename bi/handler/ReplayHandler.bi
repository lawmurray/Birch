/**
 * Abstract event handler that replays a trace of events.
 *
 * - replay: The trace to replay.
 */
class ReplayHandler < EventHandler {
  /**
   * Trace of events to replay.
   */
  replay:List<Event>?;
  
  /**
   * Discard events rather than replay them?
   */
  discard:Boolean <- false;
  
  function handle(evt:FactorEvent) -> Real {
    return evt.observe();
  }
  
  function handle(evt:RandomEvent) -> Real {
    auto replayEvt <- next();
    if evt.hasValue() {
      return evt.observe();
    } else if replayEvt? && replayEvt!.hasValue() {
      if delay {
        evt.assumeUpdate(replayEvt!);
      } else {
        evt.value(replayEvt!);
      }
    } else {
      if delay {
        evt.assume();
      } else {
        evt.value();
      }
    }
    return 0.0;
  }
  
  /**
   * Get the next event to replay, if any. If the discard flag is set, this
   * discards the next event and returns `nil`.
   */
  function next() -> Event? {
    evt:Event?;
    if replay? && !(replay!.empty()) {
      if !discard {
        evt <- replay!.front();
      }
      replay!.popFront();
    }
    return evt;
  }
  
  /**
   * Remove and return the replay trace.
   */
  function takeReplay() -> List<Event>? {
    auto replay <- this.replay;
    this.replay <- nil;
    return replay;
  }
  
  /**
   * Set the replay trace.
   */
  function setReplay(replay:List<Event>?) {
    this.replay <- replay;
  }

  /**
   * Set the discard flag. When true, events from the trace are discarded
   * rather than replayed.
   */
  function setDiscard(discard:Boolean) {
    this.discard <- discard;
  }
}
