/**
 * Abstract event triggered by the simulation of a model.
 */
class Event {
  /**
   * Accept an event handler.
   */
  function accept(h:Handler) -> Real;

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
    assert isRandom();
  }

  /**
   * For a random variate event, is there a distribution already assigned?
   */
  function hasDistribution() -> Boolean {
    assert isRandom();
  }

  /**
   * Enact assume, for a random variable event.
   */
  function assume();

  /**
   * Enact value, for a random variable event.
   */
  function value();

  /**
   * Enact simulate, for a random variable event.
   */
  function simulate();
  
  /**
   * Enact observe, for a factor or random variable event with a value.
   *
   * Returns: the log-weight associated with the event.
   */
  function observe() -> Real {
    assert isFactor() || isRandom();
  }

  /**
   * Enact update, for a random variable event with a value.
   */
  function update();

  /**
   * Enact downdate, for a random variable event with a value.
   */
  function downdate();
}
