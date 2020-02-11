/**
 * Integer value.
 */
class IntegerValue(value:Integer) < Value {
  /**
   * The value.
   */
  value:Integer <- value;
  
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
    return Real(value);
  }

  function getRealVector() -> Real[_]? {
    return vector(Real(value), 1);
  }

  function getRealMatrix() -> Real[_,_]? {
    return matrix(Real(value), 1, 1);
  }
}
