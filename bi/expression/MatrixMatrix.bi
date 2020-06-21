/**
 * Lazy `matrix`.
 */
final class MatrixMatrix<Single,Value>(x:Single) <
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
    m:MatrixMatrix<Expression<LLT>,Real[_,_]>(x);
    return m;
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
      m:MatrixMatrix<Expression<Real[_,_]>,Real[_,_]>(x);
      return m;
    }
  } else {
    return x;
  }
}
