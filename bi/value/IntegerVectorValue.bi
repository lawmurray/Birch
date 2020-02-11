/**
 * Integer vector value.
 */
class IntegerVectorValue(value:Integer[_]) < Value {
  /**
   * The value.
   */
  value:Integer[_] <- value;

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
    value:Real[_] <- this.value;
    return value;
  }

  function getRealMatrix() -> Real[_,_]? {
    value:Real[_] <- this.value;
    return column(value);
  }
}
