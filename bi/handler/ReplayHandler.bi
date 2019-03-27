/**
 * Abstract event handler that replays a trace of events.
 *
 * - replay: The trqce to replay.
 */
class ReplayHandler < Handler {
  /**
   * Trace of events to replay.
   */
  replay:List<Event>?;
  
  /**
   * Discard events rather than replay them?
   */
  discard:Boolean <- false;
  
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
