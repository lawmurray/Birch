/**
 * Boolean value.
 */
class BooleanValue(value:Boolean) < Value {
  /**
   * The value.
   */
  value:Boolean <- value;
  
  operator -> Boolean {
    return value;
  }

  function accept(writer:Writer) {
    writer.visit(this);
  }

  function isScalar() -> Boolean {
    return true;
  }
  
  function getBoolean() -> Boolean? {
    return value;
  }

  function getBooleanVector() -> Boolean[_]? {
    return vector(value, 1);
  }

  function getBooleanMatrix() -> Boolean[_,_]? {
    return matrix(value, 1, 1);
  }
}
