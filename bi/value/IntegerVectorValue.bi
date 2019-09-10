/**
 * Integer vector value.
 */
class IntegerVectorValue(value:Integer[_]) < Value {
  /**
   * The value.
   */
  value:Integer[_] <- value;
  
  operator -> Integer[_] {
    return value;
  }

  operator -> Real[_] {
    return value;
  }

  function accept(writer:Writer) {
    writer.visit(this);
  }

  function isArray() -> Boolean {
    return true;
  }
  
  function getIntegerVector() -> Integer[_]? {
    return value;
  }

  function getIntegerMatrix() -> Integer[_,_]? {
    return column(value);
  }

  function getRealVector() -> Real[_]? {
    return value;
  }

  function getRealMatrix() -> Real[_,_]? {
    return column(value);
  }
}

function IntegerVectorValue(value:Integer[_]) -> IntegerVectorValue {
  o:IntegerVectorValue(value);
  return o;
}
