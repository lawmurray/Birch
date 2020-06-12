/**
 * Lazy `transpose`.
 */
final class MatrixTranspose<Single,Value>(x:Single) <
    MatrixUnaryExpression<Single,Value>(x) {
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
    m:MatrixTranspose<Expression<Real[_,_]>,Real[_,_]>(x);
    return m;
  }
}

/**
 * Lazy `transpose`.
 */
function transpose(x:Expression<LLT>) -> Expression<LLT> {
  if x.isConstant() {
    return box(transpose(x.value()));
  } else {
    m:MatrixTranspose<Expression<LLT>,LLT>(x);
    return m;
  }
}
