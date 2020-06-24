/**
 * Lazy `diagonal`.
 */
final class MatrixDiagonal(x:Expression<Real[_]>) <
    MatrixUnaryExpression<Expression<Real[_]>,Real[_,_]>(x) {
  override function doRows() -> Integer {
    return single!.rows();
  }
  
  override function doColumns() -> Integer {
    return single!.rows();
  }

  override function doValue() {
    x <- diagonal(single!.value());
  }

  override function doPilot() {
    x <- diagonal(single!.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- diagonal(single!.move(κ));
  }

  override function doGrad() {
    single!.grad(diagonal(d!));
  }
}

/**
 * Lazy `diagonal`.
 */
function diagonal(x:Expression<Real[_]>) -> Expression<Real[_,_]> {
  if x.isConstant() {
    return box(matrix(diagonal(x.value())));
  } else {
    m:MatrixDiagonal(x);
    return m;
  }
}

