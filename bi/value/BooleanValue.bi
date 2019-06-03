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
}

function BooleanValue(value:Boolean) -> BooleanValue {
  o:BooleanValue(value);
  return o;
}
