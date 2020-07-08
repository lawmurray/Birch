/**
 * Lazy `inv`.
 */
final class MatrixInv<Single,Value>(x:Single) <
    MatrixUnaryExpression<Single,Value>(x) {
  override function doRows() -> Integer {
    return single!.rows();
  }
  
  override function doColumns() -> Integer {
    return single!.columns();
  }

  override function doValue() {
    x <- inv(single!.value());
  }

  override function doPilot() {
    x <- inv(single!.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- inv(single!.move(κ));
  }

  override function doGrad() {
    single!.grad(-transpose(x!)*d!*transpose(x!));
  }
}

/**
 * Lazy `inv`.
 */
function inv(x:Expression<Real[_,_]>) -> Expression<Real[_,_]> {
  if x.isConstant() {
    return box(matrix(inv(x.value())));
  } else {
    return construct<MatrixInv<Expression<Real[_,_]>,Real[_,_]>>(x);
  }
}

/**
 * Lazy `inv`.
 */
function inv(x:Expression<LLT>) -> Expression<LLT> {
  if x.isConstant() {
    return box(llt(inv(x.value())));
  } else {
    return construct<MatrixInv<Expression<LLT>,LLT>>(x);
  }
}
