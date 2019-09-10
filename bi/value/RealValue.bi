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

  function accept(writer:Writer) {
    writer.visit(this);
  }

  function isScalar() -> Boolean {
    return true;
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

function RealValue(value:Real) -> RealValue {
  o:RealValue(value);
  return o;
}
