/**
 * Lazy `matrix`.
 */
final class MatrixMatrix<Single,Value>(x:Single) <
    MatrixUnaryExpression<Single,Value>(x) {
  override function doRows() -> Integer {
    return single!.rows();
  }
  
  override function doColumns() -> Integer {
    return single!.columns();
  }

  override function doValue() {
    x <- matrix(single!.value());
  }

  override function doPilot() {
    x <- matrix(single!.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- matrix(single!.move(κ));
  }

  override function doGrad() {
    single!.grad(d!);
  }
}

/**
 * Lazy `matrix`.
 */
function matrix(x:Expression<LLT>) -> Expression<Real[_,_]> {
  if x.isConstant() {
    return box(matrix(x.value()));
  } else {
    return construct<MatrixMatrix<Expression<LLT>,Real[_,_]>>(x);
  }
}

/**
 * Lazy `matrix`.
 */
function matrix(x:Expression<Real[_,_]>) -> Expression<Real[_,_]> {
  if x.isRandom() {
    /* Random objects are wrapped as the accumulation of gradients by element
     * requires this; see note in split() also */
    if x.isConstant() {
      return box(matrix(x.value()));
    } else {
      return construct<MatrixMatrix<Expression<Real[_,_]>,Real[_,_]>>(x);
    }
  } else {
    return x;
  }
}
