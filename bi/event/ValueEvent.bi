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
  final function coerce(record:Record) -> Value {
    auto r <- ValueRecord<Value>?(record);
    if !r? {
      error("incompatible trace");
    }
    return r!.value();
  }

  /**
   * Coerce a random out of a record. This tries to cast the value in the
   * record to the required type and return it.
   */
  final function coerceRandom(record:Record) -> Random<Value> {
    auto r <- DelayRecord<Value>?(record);
    if !r? {
      error("incompatible trace");
    }
    return r!.random();
  }
}
