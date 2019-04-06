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
  function hasValue() -> Boolean;

  /**
   * For a random variate event, is there a distribution already assigned?
   */
  function hasDistribution() -> Boolean;

  /**
   * Enact assume, for a random event.
   */
  function assume();

  /**
   * Enact observe, for a factor or random event with a value.
   *
   * Returns: the log-weight associated with the event.
   */
  function observe() -> Real;

  /**
   * Enact value, for a random event.
   */
  function value();

  /**
   * Enact assumeUpdate, for a random event, where the value is provided by
   * another event.
   */
  function assumeUpdate(evt:Event);

  /**
   * Enact assumeDowndate, for a random event, where the value is provided by
   * another event.
   */
  function assumeDowndate(evt:Event);

  /**
   * Enact value, for a random event, where the value is provided by another
   * event.
   */
  function value(evt:Event);
}
