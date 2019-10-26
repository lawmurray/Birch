/**
 * Abstract event triggered by the simulation of a model.
 */
abstract class Event {
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
  abstract function playImmediate() -> Real;

  /**
   * Act as appropriate for `PLAY_DELAY` mode.
   */
  function playDelay() -> Real {
    return playImmediate();
  }

  /**
   * Act as appropriate for `SKIP_IMMEDIATE` mode.
   */
  abstract function skipImmediate(trace:Queue<Record>) -> Real;

  /**
   * Act as appropriate for `SKIP_DELAY` mode.
   */
  function skipDelay(trace:Queue<Record>) -> Real {
    return skipImmediate(trace);
  }
  
  /**
   * Act as appropriate for `REPLAY_IMMEDIATE` mode.
   */
  abstract function replayImmediate(trace:Queue<Record>) -> Real;

  /**
   * Act as appropriate for `REPLAY_DELAY` mode.
   */
  function replayDelay(trace:Queue<Record>) -> Real {
    return replayImmediate(trace);
  }

  /**
   * Act as appropriate for `PROPOSE_IMMEDIATE` mode.
   */
  abstract function proposeImmediate(trace:Queue<Record>) -> Real;

  /**
   * Act as appropriate for `DOWNDATE_IMMEDIATE` mode.
   */
  abstract function downdateImmediate(trace:Queue<Record>) -> Real;

  /**
   * Act as appropriate for `DOWNDATE_DELAY` mode.
   */
  function downdateDelay(trace:Queue<Record>) -> Real {
    return downdateImmediate(trace);
  }

  /**
   * Record the event in the given trace. The event is free to choose how
   * to do so, including not recording itself, if no information will be
   * required for replay.
   */
  function record(trace:Queue<Record>) {
    //
  }
}
