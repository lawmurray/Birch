/**
 * Lazy `diagonal`.
 */
final class Diagonal<Argument,Value>(x:Expression<Argument>, n:Integer) <
    UnaryExpression<Argument,Value>(x) {
  /**
   * Size.
   */
  n:Integer <- n;
  
  override function rows() -> Integer {
    return n;
  }
  
  override function columns() -> Integer {
    return n;
  }

  override function computeValue(x:Argument) -> Value {
    return diagonal(x, n);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
    return sum(diagonal(d));
  }
}

/**
 * Lazy `diagonal`.
 */
function diagonal(x:Expression<Real>, n:Integer) -> Diagonal<Real,Real[_,_]> {
  m:Diagonal<Real,Real[_,_]>(x, n);
  return m;
}
