/**
 * Integer matrix value.
 */
class IntegerMatrixValue(value:Integer[_,_]) < Value {
  /**
   * The value.
   */
  value:Integer[_,_] <- value;
  
  operator -> Integer[_,_] {
    return value;
  }

  operator -> Real[_,_] {
    return value;
  }

  function accept(writer:Writer) {
    writer.visit(this);
  }
  
  function isArray() -> Boolean {
    return true;
  }
  
  function getIntegerMatrix() -> Integer[_,_]? {
    return value;
  }

  function getRealMatrix() -> Real[_,_]? {
    return value;
  }
}
