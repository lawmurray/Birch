/**
 * Lazy `diagonal`.
 */
final class MatrixDiagonal(x:Expression<Real[_]>) <
    MatrixUnaryExpression<Expression<Real[_]>,Real[_,_]>(x) {
  override function rows() -> Integer {
    return single.rows();
  }
  
  override function columns() -> Integer {
    return single.rows();
  }

  override function doValue() {
    x <- diagonal(single.value());
  }

  override function doGet() {
    x <- diagonal(single.get());
  }

  override function doPilot() {
    x <- diagonal(single.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- diagonal(single.move(κ));
  }

  override function doGrad() {
    single.grad(diagonal(D!));
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

