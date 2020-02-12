/**
 * Lazy `column`.
 */
final class Column<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  function rows() -> Integer {
    return single.rows();
  }
  
  function columns() -> Integer {
    return 1;
  }

  function doValue(x:Argument) -> Value {
    return column(x);
  }

  function doGradient(d:Value, x:Argument) -> Argument {
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
