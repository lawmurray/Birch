class TransformAffine(a:Real, c:Real) {
  /**
   * Scale.
   */
  a:Real <- a;

  /**
   * Offset.
   */
  c:Real <- c;
  
  function multiply(x:Real) {
    a <- x*a;
    c <- x*c;
  }

  function add(x:Real) {
    c <- c + x;
  }
}
