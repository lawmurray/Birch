/**
 * Event triggered by a distribution being attached to a random variate with
 * the `~` operator.
 *
 * - v: The random variate.
 * - p: The distribution.
 */
class RandomEvent<Value>(v:Random<Value>, p:Distribution<Value>) < Event {
  /**
   * Random variable associated with the event.
   */
  v:Random<Value> <- v;
  
  /**
   * Distribution associated with the event.
   */
  p:Distribution<Value> <- p;

  function isRandom() -> Boolean {
    return true;
  }

  function hasValue() -> Boolean {
    return v.hasValue();
  }

  function assume() {
    v.assume(p);
  }

  function simulate() {
    v <- p.simulate();
  }
  
  function observe() -> Real {
    assert hasValue();
    return p.observe(v);
  }
}

/**
 * Create a RandomEvent.
 */
function RandomEvent<Value>(v:Random<Value>, p:Distribution<Value>) ->
    RandomEvent<Value> {
  evt:RandomEvent<Value>(v, p);
  return evt;
}
