/**
 * Abstract record in a trace, with a value.
 */
abstract class ValueRecord<Value> < Record {
  /**
   * Get the value.
   */
  abstract function value() -> Value;
}
