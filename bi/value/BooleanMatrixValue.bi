/**
 * Boolean matrix value.
 */
class BooleanMatrixValue(value:Boolean[_,_]) < Value {
  /**
   * The value.
   */
  value:Boolean[_,_] <- value;
  
  operator -> Boolean[_,_] {
    return value;
  }

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

function BooleanMatrixValue(value:Boolean[_,_]) -> BooleanMatrixValue {
  o:BooleanMatrixValue(value);
  return o;
}
