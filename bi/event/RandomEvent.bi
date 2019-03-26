/**
 * Abstract event triggered by a distribution being attached to a random
 * variate with the `~` operator.
 */
class RandomEvent < Event {
  function isRandom() -> Boolean {
    return true;
  }
}

/**
 * Create a RandomEvent.
 */
function RandomEvent<Value>(v:Random<Value>, p:Distribution<Value>) ->
    RandomEvent {
  evt:RandomValueEvent<Value>(v, p);
  return evt;
}
