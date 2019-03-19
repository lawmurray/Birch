/**
 * Real value.
 */
class RealValue(value:Real) < Value {
  /**
   * The value.
   */
  value:Real <- value;

  operator -> Real {
    return value;
  }

  function accept(gen:Generator) {
    gen.visit(this);
  }

  function isValue() -> Boolean {
    return true;
  }
  
  function getReal() -> Real? {
    return value;
  }
}
