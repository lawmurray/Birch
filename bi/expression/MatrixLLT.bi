/**
 * Lazy `llt`.
 */
final class MatrixLLT(x:Expression<Real[_,_]>) <
    MatrixUnaryExpression<Expression<Real[_,_]>,LLT>(x) {
  override function doRows() -> Integer {
    return single!.rows();
  }
  
  override function doColumns() -> Integer {
    return single!.columns();
  }

  override function doValue() {
    x <- llt(single!.value());
  }

  override function doPilot() {
    x <- llt(single!.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- llt(single!.move(κ));
  }

  override function doGrad() {
    /* just a factorization, so only need to pass through */
    single!.grad(d!);
  }
}

/**
 * Lazy `inv`.
 */
function llt(x:Expression<Real[_,_]>) -> Expression<LLT> {
  if x.isConstant() {
    return box(llt(x.value()));
  } else {
    return construct<MatrixLLT>(x);
  }
}
