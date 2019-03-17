/**
 * Boolean value.
 */
class BooleanValue(value:Boolean) < Value {
  /**
   * The value.
   */
  value:Boolean <- value;
  
  function getBoolean() -> Boolean? {
    return value;
  }
}
