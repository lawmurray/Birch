/**
 * Integer value.
 */
class IntegerValue(value:Integer) < Value {
  /**
   * The value.
   */
  value:Integer <- value;

  operator -> Integer {
    return value;
  }
  
  function accept(writer:Writer) {
    writer.visit(this);
  }

  function isScalar() -> Boolean {
    return true;
  }

  function getInteger() -> Integer? {
    return value;
  }

  function getIntegerVector() -> Integer[_]? {
    return vector(value, 1);
  }

  function getIntegerMatrix() -> Integer[_,_]? {
    return matrix(value, 1, 1);
  }

  function getReal() -> Real? {
    return value;
  }

  function getRealVector() -> Real[_]? {
    return vector(value, 1);
  }

  function getRealMatrix() -> Real[_,_]? {
    return matrix(value, 1, 1);
  }
}

function IntegerValue(value:Integer) -> IntegerValue {
  o:IntegerValue(value);
  return o;
}
