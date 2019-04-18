/**
 * Abstract event triggered by the simulation of a model.
 */
class Event {
  /**
   * Is this a *simulate* event?
   */
  function isSimulate() -> Boolean {
    return false;
  }

  /**
   * Is this an *observe* event?
   */
  function isObserve() -> Boolean {
    return false;
  }

  /**
   * Is this a *factor* event?
   */
  function isFactor() -> Boolean {
    return false;
  }

  /**
   * Is this an *assume* event?
   */
  function isAssume() -> Boolean {
    return false;
  }

  /**
   * Is this a *value* event?
   */
  function isValue() -> Boolean {
    return false;
  }

  /**
   * Act as appropriate for `PLAY_IMMEDIATE` mode.
   */
  function playImmediate() -> Real {
    assert false;
  }

  /**
   * Act as appropriate for `PLAY_DELAY` mode.
   */
  function playDelay() -> Real {
    return playImmediate();
  }
  
  /**
   * Act as appropriate for `REPLAY_IMMEDIATE` mode.
   */
  function replayImmediate(trace:Queue<Event>) -> Real {
    assert false;
  }

  /**
   * Act as appropriate for `REPLAY_DELAY` mode.
   */
  function replayDelay(trace:Queue<Event>) -> Real {
    return replayImmediate(trace);
  }

  /**
   * Act as appropriate for `DOWNDATE_IMMEDIATE` mode.
   */
  function downdateImmediate(trace:Queue<Event>) -> Real {
    assert false;
  }

  /**
   * Act as appropriate for `DOWNDATE_DELAY` mode.
   */
  function downdateDelay(trace:Queue<Event>) -> Real {
    return downdateImmediate(trace);
  }

  /**
   * Record the event in the given trace. The event is free to choose how
   * to do so, including not recording itself, if no information will be
   * required for replay.
   */
  function record(trace:Queue<Event>) {
    //
  }
}
