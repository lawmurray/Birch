/*
 * Linear transformation.
 */
class TransformLinear<Value>(a:Real, x:Value, c:Real) {
  /**
   * Scale.
   */
  a:Real <- a;

  /**
   * Delay node.
   */
  x:Value <- x;

  /**
   * Offset.
   */
  c:Real <- c;
  
  function multiply(y:Real) {
    a <- y*a;
    c <- y*c;
  }

  function divide(y:Real) {
    a <- a/y;
    c <- c/y;
  }

  function add(y:Real) {
    c <- c + y;
  }

  function subtract(y:Real) {
    c <- c - y;
  }
  
  function negateAndAdd(y:Real) {
    a <- -a;
    c <- y - c;
  }
}

function TransformLinear<Value>(a:Real, x:Value, c:Real) ->
    TransformLinear<Value> {
  m:TransformLinear<Value>(a, x, c);
  return m;
}

function TransformLinear<Value>(a:Real, x:Value) -> TransformLinear<Value> {
  return TransformLinear<Value>(a, x, 0.0);
}
