/**
 * Lazy `diagonal`.
 */
final class Diagonal<Argument,Value>(x:Expression<Argument>, n:Integer) <
    UnaryExpression<Argument,Value>(x) {
  /**
   * Size.
   */
  auto n <- n;
  
  function rows() -> Integer {
    return n;
  }
  
  function columns() -> Integer {
    return n;
  }

  function graft() -> Expression<Value> {
    return diagonal(single.graft(), n);
  }
  
  function doValue(x:Argument) -> Value {
    return diagonal(x, n);
  }

  function doGradient(d:Value, x:Argument) -> Argument {
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
