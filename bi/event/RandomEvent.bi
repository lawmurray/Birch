/**
 * Event triggered for a random variable.
 *
 * - v: The random variable.
 */
class RandomEvent<Value>(v:Random<Value>) < Event {
  /**
   * Random variable associated with the event.
   */
  v:Random<Value> <- v;

  function isRandom() -> Boolean {
    return true;
  }

  function hasValue() -> Boolean {
    return v.hasValue();
  }

  function simulate() {
    v.simulate();
  }
  
  function observe() -> Real {
    assert hasValue();
    return v.observe();
  }

  function update() {
    assert hasValue();
    return v.update();
  }

  function downdate() {
    assert hasValue();
    return v.downdate();
  }
}

/**
 * Create a RandomEvent.
 */
function RandomEvent<Value>(v:Random<Value>) -> RandomEvent<Value> {
  evt:RandomEvent<Value>(v);
  return evt;
}
