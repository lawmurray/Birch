/**
 * Boolean vector value.
 */
class BooleanVectorValue(value:Boolean[_]) < Value {
  /**
   * The value.
   */
  value:Boolean[_] <- value;
  
  operator -> Boolean[_] {
    return value;
  }

  function accept(writer:Writer) {
    writer.visit(this);
  }

  function isArray() -> Boolean {
    return true;
  }
  
  function getBooleanVector() -> Boolean[_]? {
    return value;
  }

  function getBooleanMatrix() -> Boolean[_,_]? {
    return column(value);
  }
}

function BooleanVectorValue(value:Boolean[_]) -> BooleanVectorValue {
  o:BooleanVectorValue(value);
  return o;
}
