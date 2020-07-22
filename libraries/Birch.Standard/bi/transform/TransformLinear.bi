/*
 * Linear transformation of a random variate, as represented by its
 * associated distribution.
 *
 * - Value: Distribution type.
 */
class TransformLinear<Value>(a:Expression<Real>, x:Value,
    c:Expression<Real>) {
  /**
   * Scale.
   */
  a:Expression<Real> <- a;

  /**
   * Distribution.
   */
  x:Value <- x;

  /**
   * Offset.
   */
  c:Expression<Real> <- c;
  
  function multiply(y:Expression<Real>) {
    a <- y*a;
    c <- y*c;
  }

  function divide(y:Expression<Real>) {
    a <- a/y;
    c <- c/y;
  }

  function add(y:Expression<Real>) {
    c <- c + y;
  }

  function subtract(y:Expression<Real>) {
    c <- c - y;
  }

  function negate() {
    a <- -a;
    c <- -c;
  }
  
  function negateAndAdd(y:Expression<Real>) {
    a <- -a;
    c <- y - c;
  }
}

function TransformLinear<Value>(a:Expression<Real>, x:Value,
    c:Expression<Real>) -> TransformLinear<Value> {
  return construct<TransformLinear<Value>>(a, x, c);
}

function TransformLinear<Value>(a:Expression<Real>, x:Value) ->
    TransformLinear<Value> {
  return TransformLinear<Value>(a, x, box(0.0));
}
