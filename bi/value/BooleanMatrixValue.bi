/**
 * Boolean matrix value.
 */
class BooleanMatrixValue(value:Boolean[_,_]) < Value {
  /**
   * The value.
   */
  value:Boolean[_,_] <- value;

  function accept(writer:Writer) {
    writer.visit(this);
  }

  function isArray() -> Boolean {
    return true;
  }
  
  function getBooleanMatrix() -> Boolean[_,_]? {
    return value;
  }
}
