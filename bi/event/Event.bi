/**
 * Abstract event triggered by the simulation of a model.
 */
class Event {
  /**
   * Accept an event handler.
   */
  function accept(h:EventHandler) -> Real;

  /**
   * Is this a factor event?
   */
  function isFactor() -> Boolean {
    return false;
  }
  
  /**
   * Is this a random variable event?
   */
  function isRandom() -> Boolean {
    return false;
  }
  
  /**
   * For a random variate event, is there a value already assigned?
   */
  function hasValue() -> Boolean {
    assert false;
  }

  /**
   * For a random variate event, is there a distribution already assigned?
   */
  function hasDistribution() -> Boolean {
    assert false;
  }

  /**
   * Enact assume, for a random event.
   */
  function assume() {
    assert false;
  }

  /**
   * Enact observe, for a factor or random event with a value.
   *
   * Returns: the log-weight associated with the event.
   */
  function observe() -> Real {
    assert false;
  }

  /**
   * Enact value, for a random event.
   */
  function value() {
    assert false;
  }

  /**
   * Enact downdate, for a random event, where the value is provided by
   * another event.
   */
  function downdate(evt:Event) {
    assert false;
  }

  /**
   * Enact assumeUpdate, for a random event, where the value is provided by
   * another event.
   */
  function assumeUpdate(evt:Event) {
    assert false;
  }

  /**
   * Enact assumeDowndate, for a random event, where the value is provided by
   * another event.
   */
  function assumeDowndate(evt:Event) {
    assert false;
  }

  /**
   * Enact value, for a random event, where the value is provided by another
   * event.
   */
  function value(evt:Event) {
    assert false;
  }
}
