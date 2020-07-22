/**
 * String value.
 */
class StringValue(value:String) < Value {
  /**
   * The value.
   */
  value:String <- value;

  function accept(writer:Writer) {
    writer.visit(this);
  }

  function isScalar() -> Boolean {
    return true;
  }

  function getString() -> String? {
    return value;
  }
}
