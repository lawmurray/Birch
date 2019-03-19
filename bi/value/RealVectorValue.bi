/**
 * Real vector value.
 */
class RealVectorValue(value:Real[_]) < Value {
  /**
   * The value.
   */
  value:Real[_] <- value;
  
  operator -> Real[_] {
    return value;
  }

  function accept(gen:Generator) {
    gen.visit(this);
  }

  function isValue() -> Boolean {
    return true;
  }
  
  function getRealVector() -> Real[_]? {
    return value;
  }
}
