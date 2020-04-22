/**
 * Lazy `column`.
 */
final class Column<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function rows() -> Integer {
    return single.rows();
  }
  
  override function columns() -> Integer {
    return 1;
  }

  override function computeValue(x:Argument) -> Value {
    return column(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
    return vector(d);
  }
}

/**
 * Lazy `column`.
 */
function column(x:Expression<Real[_]>) -> Column<Real[_],Real[_,_]> {
  m:Column<Real[_],Real[_,_]>(x);
  return m;
}
