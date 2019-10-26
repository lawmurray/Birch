/**
 * Abstract event, with a value.
 */
abstract class ValueEvent<Value> < Event {
  /**
   * Does this have a value?
   */
  abstract function hasValue() -> Boolean;

  /**
   * Get the value.
   */
  abstract function value() -> Value;
}
