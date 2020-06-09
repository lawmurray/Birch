/**
 * Lazy `transpose`.
 */
final class MatrixTranspose(x:Expression<Real[_,_]>) <
    MatrixUnaryExpression<Expression<Real[_,_]>,Real[_,_]>(x) {
  override function rows() -> Integer {
    return single.columns();
  }
  
  override function columns() -> Integer {
    return single.rows();
  }

  override function doValue() {
    x <- transpose(single.value());
  }

  override function doGet() {
    x <- transpose(single.get());
  }

  override function doPilot() {
    x <- transpose(single.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- transpose(single.move(κ));
  }

  override function doGrad() {
    single.grad(transpose(D!));
  }
}

/**
 * Lazy `transpose`.
 */
function transpose(x:Expression<Real[_,_]>) -> Expression<Real[_,_]> {
  if x.isConstant() {
    return box(matrix(transpose(x.value())));
  } else {
    m:MatrixTranspose(x);
    return m;
  }
}
