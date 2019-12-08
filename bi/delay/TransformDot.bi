/*
 * Multivariate linear transformation involving a dot product.
 */
class TransformDot<Value>(a:Real[_], x:Value, c:Real) {
  /**
   * Scale.
   */
  auto a <- a;
  
  /**
   * Delay node.
   */
  auto x <- x;

  /**
   * Offset.
   */
  auto c <- c;
   
  function multiply(y:Real) {
    a <- a*y;
    c <- c*y;
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

function TransformDot<Value>(a:Real[_], x:Value, c:Real) ->
    TransformDot<Value> {
  m:TransformDot<Value>(a, x, c);
  return m;
}

function TransformDot<Value>(a:Real[_], x:Value) -> TransformDot<Value> {
  return TransformDot<Value>(a, x, 0.0);
}
