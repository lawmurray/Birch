/**
 * String value.
 */
class StringValue(value:String) < Value {
  /**
   * The value.
   */
  value:String <- value;

  operator -> String {
    return value;
  }

  function accept(gen:Generator) {
    gen.visit(this);
  }

  function isValue() -> Boolean {
    return true;
  }

  function getString() -> String? {
    return value;
  }
}

function StringValue(value:String) -> StringValue {
  o:StringValue(value);
  return o;
}
