/**
 * Abstract event handler.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
class Handler {
  /**
   * Handle a sequence of events.
   *
   * Returns: Log-weight.
   */
  function handle(evt:Event!) -> Real {
    auto w <- 0.0;
    while evt? {
      w <- w + evt!.accept(this);
    }
    return w;
  }

  /**
   * Handle a factor event.
   *
   * - evt: The event.
   *
   * Returns: Log-weight.
   */
  function handle(evt:FactorEvent) -> Real;
  
  /**
   * Handle a random event.
   *
   * - evt: The event.
   *
   * Returns: Log-weight.
   */
  function handle(evt:RandomEvent) -> Real;

  /**
   * If this is a replay event handler, clear and return the replay trace.
   */
  function takeReplay() -> List<Event>? {
    assert false;
  }
  
  /**
   * If this is a replay event handler, set the replay trace.
   */
  function setReplay(replay:List<Event>?) {
    assert false;
  }

  /**
   * If this is a trace event handler, clear and return the recorded trace.
   */
  function takeRecord() -> List<Event> {
    assert false;
  }
  
  /**
   * If this is a trace event handler, set the recorded trace.
   */
  function setRecord(replay:List<Event>) {
    assert false;
  }

  /**
   * Convenience function equivalent to `this.setReplay(this.takeRecord())`.
   */
  function rewind() {
    setReplay(takeRecord());
  }

  /**
   * If this is a replay event handler, set the discard flag.
   */
  function setDiscard(discard:Boolean) {
    assert false;
  }
}
