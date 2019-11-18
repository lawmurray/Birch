/**
 * Abstract record in a trace.
 */
abstract class Record {
  /**
   * Does this have a value?
   */
  abstract function hasValue() -> Boolean;
}
