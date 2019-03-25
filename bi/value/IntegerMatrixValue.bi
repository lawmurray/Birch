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

  function accept(gen:Generator) {
    gen.visit(this);
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

function IntegerMatrixValue(value:Integer[_,_]) -> IntegerMatrixValue {
  o:IntegerMatrixValue(value);
  return o;
}
