/**
 * String value.
 */
class StringValue(value:String) < Value {
  /**
   * The value.
   */
  value:String <- value;

  function getString() -> String? {
    return value;
  }
}
