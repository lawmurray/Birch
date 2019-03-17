/**
 * Integer value.
 */
class IntegerValue(value:Integer) < Value {
  /**
   * The value.
   */
  value:Integer <- value;
  
  function getInteger() -> Integer? {
    return value;
  }

  function getReal() -> Real? {
    return value;
  }
}
