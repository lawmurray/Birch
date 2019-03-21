/**
 * Abstract event triggered by the simulation of a model.
 */
class Event {
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
   * For a random variable event, is there a variate already assigned to the
   * random variable.
   */
  function hasValue() -> Boolean {
    assert isRandom();
  }

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
