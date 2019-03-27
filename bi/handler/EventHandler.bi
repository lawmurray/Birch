/**
 * Event handler.
 *
 * The EventHandler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 *
 * Two different modes are supported. In *immediate* mode:
 *
 *   * random events with values trigger observation, and
 *   * random events without values trigger simulation,
 *
 * while in *delayed* mode:
 *
 *   * random events with values trigger observation, but
 *   * random events without values do not trigger immediate simulation, but
 *     are instead prepared for simulation on-demand, or variable
 *     elimination.
 *
 * Delayed mode corresponds to using delayed sampling.
 */
class EventHandler {
  /**
   * Delayed sampling flag.
   */
  delay:Boolean <- true;
  
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
  function handle(evt:FactorEvent) -> Real {
    return evt.observe();
  }

  /**
   * Handle a random event.
   *
   * - evt: The event.
   *
   * Returns: Log-weight.
   */  
  function handle(evt:RandomEvent) -> Real {
    if evt.hasValue() {
      return evt.observe();
    } else if (delay) {
      evt.assume();
    } else {    
      evt.value();
    }
    return 0.0;
  }

  /**
   * Set the delayed sampling flag.
   */
  function setDelay(delay:Boolean) {
    this.delay <- delay;
  }

  /**
   * If this is a replay event handler, set the discard flag.
   */
  function setDiscard(discard:Boolean) {
    assert false;
  }

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
}
