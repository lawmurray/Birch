/**
 * Real value.
 */
class RealValue(value:Real) < Value {
  /**
   * The value.
   */
  value:Real <- value;
  
  function getReal() -> Real? {
    return value;
  }
}
