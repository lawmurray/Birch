/*
 * Delayed expression.
 *
 * - Value: Value type.
 */
class Expression<Value> < Delay {  
  /**
   * Value conversion.
   */
  operator -> Value {
    return value();
  }
  
  /**
   * Value conversion.
   */
  function value() -> Value;
  
  /**
   * Are the values of any random variables upon this expression depends
   * missing?
   */
  function isMissing() -> Boolean;
}
