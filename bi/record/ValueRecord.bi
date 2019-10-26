/**
 * Abstract record in a trace, with a value.
 */
abstract class ValueRecord<Value> < Record {
  /**
   * Does this have a value?
   */
  abstract function hasValue() -> Boolean;

  /**
   * Get the value.
   */
  abstract function value() -> Value;
}
