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

  function accept(writer:Writer) {
    writer.visit(this);
  }

  function isArray() -> Boolean {
    return true;
  }

  function getRealVector() -> Real[_]? {
    return value;
  }

  function getRealMatrix() -> Real[_,_]? {
    return column(value);
  }
}

function RealVectorValue(value:Real[_]) -> RealVectorValue {
  o:RealVectorValue(value);
  return o;
}
