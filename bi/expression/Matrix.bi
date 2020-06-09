/**
 * Lazy `matrix`.
 */
final class Matrix<Single,Value>(x:Single) <
    MatrixUnaryExpression<Single,Value>(x) {
  override function rows() -> Integer {
    return single.rows();
  }
  
  override function columns() -> Integer {
    return single.columns();
  }

  override function doValue() {
    x <- matrix(single.value());
  }

  override function doGet() {
    x <- matrix(single.get());
  }

  override function doPilot() {
    x <- matrix(single.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- matrix(single.move(κ));
  }

  override function doGrad() {
    single.grad(D!);
  }
}

/**
 * Lazy `matrix`.
 */
function matrix(x:Expression<LLT>) -> Expression<Real[_,_]> {
  if x.isConstant() {
    return box(matrix(x.value()));
  } else {
    m:Matrix<Expression<LLT>,Real[_,_]>(x);
    return m;
  }
}

/**
 * Lazy `matrix`.
 */
function matrix(x:Expression<Real[_,_]>) -> Expression<Real[_,_]> {
  return x;
}
