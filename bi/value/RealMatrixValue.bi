/**
 * Real matrix value.
 */
class RealMatrixValue(value:Real[_,_]) < Value {
  /**
   * The value.
   */
  value:Real[_,_] <- value;
  
  operator -> Real[_,_] {
    return value;
  }

  function accept(writer:Writer) {
    writer.visit(this);
  }
  
  function getRealMatrix() -> Real[_,_]? {
    return value;
  }

  function isArray() -> Boolean {
    return true;
  }
}
