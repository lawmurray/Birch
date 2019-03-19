/**
 * Boolean matrix value.
 */
class BooleanMatrixValue(value:Boolean[_,_]) < Value {
  /**
   * The value.
   */
  value:Boolean[_,_] <- value;
  
  operator -> Boolean[_,_] {
    return value;
  }

  function accept(gen:Generator) {
    gen.visit(this);
  }

  function isValue() -> Boolean {
    return true;
  }
  
  function getBooleanMatrix() -> Boolean[_,_]? {
    return value;
  }
}
