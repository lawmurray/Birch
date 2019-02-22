/*
 * Multivariate linear transformation.
 */
class TransformMultivariateDot<Value>(a:Value[_], c:Value) {
  /**
   * Scale.
   */
  a:Value[_] <- a;

  /**
   * Offset.
   */
  c:Value <- c;
    
  function add(x:Value) {
    c <- c + x;
  }

  function subtract(x:Value) {
    c <- c - x;
  }

  function multiply(x:Value) {
    a <- a*x;
    c <- c*x;
  }

  function divide(x:Value) {
    a <- a/x;
    c <- c/x;
  }
  
  function negateAndAdd(x:Value) {
    a <- -a;
    c <- x - c;
  }
}
