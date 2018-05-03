/*
 * Linear transformation of a variate.
 */
class TransformLinear<Value>(a:Value, c:Value) {
  /**
   * Scale.
   */
  a:Value <- a;

  /**
   * Offset.
   */
  c:Value <- c;
  
  function multiply(x:Value) {
    a <- x*a;
    c <- x*c;
  }

  function add(x:Value) {
    c <- c + x;
  }

  function subtract(x:Value) {
    c <- c - x;
  }
  
  function negateAndAdd(x:Value) {
    a <- -a;
    c <- x - c;
  }
}
