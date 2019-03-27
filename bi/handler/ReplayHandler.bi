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
   * Get the next event to replay, if any.
   */
  function next() -> Event? {
    evt:Event?;
    if replay? && !(replay!.empty()) {
      evt <- replay!.front();
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
}
