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

  /**
   * Coerce a value out of a record. This tries to cast the value in the
   * record to the required type and return it.
   */
  function coerce(record:Record) -> Value {
    auto r <- ValueRecord<Value>?(record);
    if !r? {
      error("incompatible trace");
    } else {
      return r!.value();
    }
  }
}
