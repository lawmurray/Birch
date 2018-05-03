class TransformMultivariateLinear(A:Real[_,_], c:Real[_]) {
  /**
   * Scale.
   */
  A:Real[_,_] <- A;

  /**
   * Offset.
   */
  c:Real[_] <- c;
  
  function size() -> Integer {
    return length(c);
  }
  
  function leftMultiply(X:Real[_,_]) {
    A <- X*A;
    c <- X*c;
  }

  function add(x:Real[_]) {
    c <- c + x;
  }
}
